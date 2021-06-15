import argparse
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from demo import networks
from demo.utils import *
from lap.renderer.renderer_mr import Renderer


EPS = 1e-7


class Demo():
    def __init__(self, args):
        ## configs
        self.device = 'cuda:0' if args.gpu else 'cpu'
        self.checkpoint_path_lap = args.checkpoint_lap
        self.detect_human_face = args.detect_human_face
        self.render_video = args.render_video
        self.output_size = args.output_size
        self.image_size_lap = 128
        self.min_depth = 0.9
        self.max_depth = 1.1
        self.border_depth = 1.05
        self.xyz_rotation_range = 60
        self.xy_translation_range = 0.1
        self.z_translation_range = 0
        self.fov = 10  # in degrees
        self.inner_batch = 1
        self.count = 0

        self.depth_rescaler = lambda d : (1+d)/2 *self.max_depth + (1-d)/2 *self.min_depth  # (-1,1) => (min_depth,max_depth)
        self.depth_inv_rescaler = lambda d :  (d-self.min_depth) / (self.max_depth-self.min_depth)  # (min_depth,max_depth) => (0,1)

        fx = (self.image_size_lap-1)/2/(np.tan(self.fov/2 *np.pi/180))
        fy = (self.image_size_lap-1)/2/(np.tan(self.fov/2 *np.pi/180))
        cx = (self.image_size_lap-1)/2
        cy = (self.image_size_lap-1)/2
        K = [[fx, 0., cx],
             [0., fy, cy],
             [0., 0., 1.]]
        K = torch.FloatTensor(K).to(self.device)
        self.inv_K_lap = torch.inverse(K).unsqueeze(0)
        self.K = K.unsqueeze(0)

        cfgs = {
                'device': self.device,
                'min_depth': self.min_depth,
                'max_depth': self.max_depth,
                'fov': self.fov,
            }
        self.renderer_mr = Renderer(cfgs, im_size=128)
        ## NN models
        self.network_names = ['netD', 'netA', 'netL', 'netV', 'refine_netA', 'refine_netD']
        self.netD = networks.ED_Aggregation(cin=3, cout=1, nf=64, zdim=512, activation=None, inner_batch=self.inner_batch, count=self.count)
        self.netA = networks.ED_Aggregation(cin=3, cout=3, nf=64, zdim=512, inner_batch=self.inner_batch, count=self.count)
        self.refine_netA = networks.ED_attribute_refining(cin=3, cout=3, nf=64, zdim=512)
        self.refine_netD = networks.ED_attribute_refining(cin=1, cout=1, nf=64, zdim=512, activation=None)
        self.netL = networks.Encoder(cin=3, cout=4, nf=32)
        self.netV = networks.Encoder(cin=3, cout=6, nf=32)

        self.netD = self.netD.to(self.device)
        self.netA = self.netA.to(self.device)
        self.refine_netA = self.refine_netA.to(self.device)
        self.refine_netD = self.refine_netD.to(self.device)
        self.netL = self.netL.to(self.device)
        self.netV = self.netV.to(self.device)
        self.load_checkpoint()
        #self.save_checkpoint()

        self.netD.eval()
        self.netA.eval()
        self.refine_netA.eval()
        self.refine_netD.eval()

        ## face detecter
        if self.detect_human_face:
            from facenet_pytorch import MTCNN
            self.face_detector = MTCNN(select_largest=True, device=self.device)

        ## renderer
        if self.render_video:
            #from lap.renderer.renderer_mr import Renderer
            assert 'cuda' in self.device, 'A GPU device is required for rendering because the neural_renderer only has GPU implementation.'
            cfgs = {
                'device': self.device,
                'image_size': self.output_size,
                'min_depth': self.min_depth,
                'max_depth': self.max_depth,
                'fov': self.fov,
            }
            self.renderer_mr = Renderer(cfgs, im_size=128)

    def load_checkpoint(self):
        print(f"Loading checkpoint from {self.checkpoint_path_lap}")
        cp_lap = torch.load(self.checkpoint_path_lap, map_location=self.device)

        self.netD.load_state_dict(cp_lap['netD'])
        self.netA.load_state_dict(cp_lap['netA'])
        self.refine_netD.load_state_dict(cp_lap['refine_netD'])
        self.refine_netA.load_state_dict(cp_lap['refine_netA'])
        #self.finer_netD.load_state_dict(cp_lap['finer_netD'])
        #self.finer_netA.load_state_dict(cp_lap['finer_netA'])
        self.netL.load_state_dict(cp_lap['netL'])
        self.netV.load_state_dict(cp_lap['netV'])

    def save_checkpoint(self):
        states = {}
        for net_name in self.network_names:
            states[net_name] = getattr(self, net_name).state_dict()
        checkpoint_path = os.path.join('./demo', 'checkpoint300_fix2.pth')
        print(f"Saving checkpoint to {checkpoint_path}")
        torch.save(states, checkpoint_path)

    def depth_to_3d_grid(self, depth, inv_K=None):
        if inv_K is None:
            inv_K = self.inv_K
        b, h, w = depth.shape
        grid_2d = get_grid(b, h, w, normalize=False).to(depth.device)  # Nxhxwx2
        depth = depth.unsqueeze(-1)
        grid_3d = torch.cat((grid_2d, torch.ones_like(depth)), dim=3)
        grid_3d = grid_3d.matmul(inv_K.transpose(2,1)) * depth
        return grid_3d
    
    def get_normal_from_depth(self, depth, inv_K=None):
        b, h, w = depth.shape
        grid_3d = self.depth_to_3d_grid(depth, inv_K=inv_K)

        tu = grid_3d[:,1:-1,2:] - grid_3d[:,1:-1,:-2]
        tv = grid_3d[:,2:,1:-1] - grid_3d[:,:-2,1:-1]
        normal = tu.cross(tv, dim=3)

        zero = normal.new_tensor([0,0,1])
        normal = torch.cat([zero.repeat(b,h-2,1,1), normal, zero.repeat(b,h-2,1,1)], 2)
        normal = torch.cat([zero.repeat(b,1,w,1), normal, zero.repeat(b,1,w,1)], 1)
        normal = normal / (((normal**2).sum(3, keepdim=True))**0.5 + EPS)
        return normal

    def detect_face(self, im):
        print("Detecting face using MTCNN face detector")
        try:
            bboxes, prob = self.face_detector.detect(im)
            w0, h0, w1, h1 = bboxes[0]
        except:
            print("Could not detect faces in the image")
            return None

        hc, wc = (h0+h1)/2, (w0+w1)/2
        crop = int(((h1-h0) + (w1-w0)) /2/2 *1.1)
        im = np.pad(im, ((crop,crop),(crop,crop),(0,0)), mode='edge')  # allow cropping outside by replicating borders
        h0 = int(hc-crop+crop + crop*0.15)
        w0 = int(wc-crop+crop)
        return im[h0:h0+crop*2, w0:w0+crop*2]

    def run(self, pil_im):
        im = np.uint8(pil_im)

        ## face detection
        if self.detect_human_face:
            im = self.detect_face(im)
            if im is None:
                return -1

        h, w, _ = im.shape
        #print(im.shape)
        im = torch.FloatTensor(im /255.).permute(2,0,1).unsqueeze(0)
        # resize to 128 first if too large, to avoid bilinear downsampling artifacts
        if h > self.image_size_lap*4 and w > self.image_size_lap*4:
            im = nn.functional.interpolate(im, (self.image_size_lap*2, self.image_size_lap*2), mode='bilinear', align_corners=False)
        im = nn.functional.interpolate(im, (self.image_size_lap, self.image_size_lap), mode='bilinear', align_corners=False)
        im_low = nn.functional.interpolate(im, (64, 64), mode='bilinear', align_corners=False)

        with torch.no_grad():
            self.input_im = im_low.to(self.device) *2.-1.
            self.input_im_lap = im.to(self.device) *2.-1.
            b, c, h, w = self.input_im.shape
            b, c_lap, h_lap, w_lap = self.input_im_lap.shape
            input_half = nn.functional.interpolate(self.input_im_lap, (128, 128), mode='bilinear', align_corners=False)

            ## predict canonical depth
            self.canon_depth_raw = self.netD(self.input_im_lap, 1)
            self.canon_depth_raw = self.refine_netD(self.canon_depth_raw, input_half)

            self.canon_depth_raw = self.canon_depth_raw.squeeze(1)
            self.canon_depth_lap = self.canon_depth_raw - self.canon_depth_raw.view(b,-1).mean(1).view(b,1,1)
            self.canon_depth_lap = self.canon_depth_lap.tanh()
            self.canon_depth_lap = self.depth_rescaler(self.canon_depth_lap)

            ## clamp border depth
            depth_border_lap = torch.zeros(1,h_lap,w_lap-4).to(self.input_im.device)
            depth_border_lap = nn.functional.pad(depth_border_lap, (2,2), mode='constant', value=1)
            self.canon_depth_lap = self.canon_depth_lap*(1-depth_border_lap) + depth_border_lap *self.border_depth
            #print(self.canon_depth_lap[:,-40:,-40:], self.canon_depth_lap[:,-40:,:40])
            self.canon_depth_lap[:,-20:,:15] = self.border_depth
            #self.canon_depth_lap[:,-5:,:] = self.border_depth

            ## predict canonical albedo
            self.canon_albedo_lap = self.netA(self.input_im_lap, 1)
            self.canon_albedo_lap = self.refine_netA(self.canon_albedo_lap, input_half)

            ## predict lighting
            canon_light = self.netL(self.input_im_lap)  # Bx4
            self.canon_light_a_lap = canon_light[:,:1] /2+0.5  # ambience term
            self.canon_light_b_lap = canon_light[:,1:2] /2+0.5  # diffuse term
            canon_light_dxy = canon_light[:,2:]
            self.canon_light_d_lap = torch.cat([canon_light_dxy, torch.ones(b,1).to(self.input_im.device)], 1)
            self.canon_light_d_lap = self.canon_light_d_lap / ((self.canon_light_d_lap**2).sum(1, keepdim=True))**0.5  # diffuse light direction

            ## shading
            self.canon_normal_lap = self.get_normal_from_depth(self.canon_depth_lap, inv_K=self.inv_K_lap)
            self.canon_diffuse_shading_lap = (self.canon_normal_lap * self.canon_light_d_lap.view(-1,1,1,3)).sum(3).clamp(min=0).unsqueeze(1)
            canon_shading = self.canon_light_a_lap.view(-1,1,1,1) + self.canon_light_b_lap.view(-1,1,1,1)*self.canon_diffuse_shading_lap
            self.canon_im_lap = (self.canon_albedo_lap/2+0.5) * canon_shading *2-1

            ## predict viewpoint transformation
            self.view_lap = self.netV(self.input_im_lap)
            self.view_lap = torch.cat([
                self.view_lap[:,:3] *np.pi/180 *self.xyz_rotation_range,
                self.view_lap[:,3:5] *self.xy_translation_range,
                self.view_lap[:,5:] *self.z_translation_range], 1)

            ## reconstruction to the target viewpoint
            self.renderer_mr.set_transform_matrices(self.view_lap)
            self.recon_depth = self.renderer_mr.warp_canon_depth(self.canon_depth_lap.to(self.device))
            self.recon_normal = self.renderer_mr.get_normal_from_depth(self.recon_depth)
            grid_2d_from_canon = self.renderer_mr.get_inv_warped_2d_grid(self.recon_depth)
            recon_im = nn.functional.grid_sample(self.canon_im_lap, grid_2d_from_canon, mode='bilinear')

            margin = (self.max_depth - self.min_depth) /2
            recon_im_mask = (self.recon_depth < self.max_depth+margin).float()  # invalid border pixels have been clamped at max_depth+margin
            recon_im_mask_both = recon_im_mask[:b] * recon_im_mask[b:]  # both original and flip reconstruction
            self.recon_im_mask_both = recon_im_mask_both.repeat(2,1,1).unsqueeze(1).detach()
            self.recon_im = recon_im * self.recon_im_mask_both

            ## export to obj strings
            vertices_lap = self.depth_to_3d_grid(self.canon_depth_lap, self.inv_K_lap)  # BxHxWx3
            self.objs_lap, self.mtls_lap = export_to_obj_string(vertices_lap, self.canon_normal_lap)

            ## resize to output size        
            self.canon_depth_lap = nn.functional.interpolate(self.canon_depth_lap.unsqueeze(1), (self.output_size, self.output_size), mode='bilinear', align_corners=False).squeeze(1)
            self.recon_depth = nn.functional.interpolate(self.recon_depth.unsqueeze(1), (self.output_size, self.output_size), mode='bilinear', align_corners=False).squeeze(1)
            self.canon_normal_lap = nn.functional.interpolate(self.canon_normal_lap.permute(0,3,1,2), (self.output_size, self.output_size), mode='bilinear', align_corners=False).permute(0,2,3,1)
            self.canon_normal_lap = self.canon_normal_lap / (self.canon_normal_lap**2).sum(3, keepdim=True)**0.5
            self.recon_normal = nn.functional.interpolate(self.recon_normal.permute(0,3,1,2), (self.output_size, self.output_size), mode='bilinear', align_corners=False).permute(0,2,3,1)
            self.recon_normal = self.recon_normal / (self.recon_normal**2).sum(3, keepdim=True)**0.5
            self.canon_diffuse_shading_lap = nn.functional.interpolate(self.canon_diffuse_shading_lap, (self.output_size, self.output_size), mode='bilinear', align_corners=False)
            self.canon_albedo_lap = nn.functional.interpolate(self.canon_albedo_lap, (self.output_size, self.output_size), mode='bilinear', align_corners=False)
            self.canon_im_lap = nn.functional.interpolate(self.canon_im_lap, (self.output_size, self.output_size), mode='bilinear', align_corners=False)

            if self.render_video:
                self.render_animation()

    def render_animation(self):
        print(f"Rendering video animations")
        b, h, w = self.canon_depth_lap.shape

        ## morph from target view to canonical
        morph_frames = 15
        view_zero = torch.FloatTensor([0.15*np.pi/180*60, 0,0,0,0,0]).to(self.canon_depth_lap.device)
        morph_s = torch.linspace(0, 1, morph_frames).to(self.canon_depth_lap.device)
        view_morph = morph_s.view(-1,1,1) * view_zero.view(1,1,-1) + (1-morph_s.view(-1,1,1)) * self.view_lap.unsqueeze(0)  # TxBx6

        ## yaw from canonical to both sides
        yaw_frames = 80
        yaw_rotations = np.linspace(-np.pi/2, np.pi/2, yaw_frames)
        # yaw_rotations = np.concatenate([yaw_rotations[40:], yaw_rotations[::-1], yaw_rotations[:40]], 0)

        ## whole rotation sequence
        view_after = torch.cat([view_morph, view_zero.repeat(yaw_frames, b, 1)], 0)
        yaw_rotations = np.concatenate([np.zeros(morph_frames), yaw_rotations], 0)

        def rearrange_frames(frames):
            morph_seq = frames[:, :morph_frames]
            yaw_seq = frames[:, morph_frames:]
            out_seq = torch.cat([
                morph_seq[:,:1].repeat(1,5,1,1,1),
                morph_seq,
                morph_seq[:,-1:].repeat(1,5,1,1,1),
                yaw_seq[:, yaw_frames//2:],
                yaw_seq.flip(1),
                yaw_seq[:, :yaw_frames//2],
                morph_seq[:,-1:].repeat(1,5,1,1,1),
                morph_seq.flip(1),
                morph_seq[:,:1].repeat(1,5,1,1,1),
            ], 1)
            return out_seq

        ## textureless shape
        front_light = torch.FloatTensor([0,0,1]).to(self.canon_depth_lap.device)
        
        canon_shape_im = (self.canon_normal_lap * front_light.view(1,1,1,3)).sum(3).clamp(min=0).unsqueeze(1)
        canon_shape_im = canon_shape_im.repeat(1,3,1,1) *0.7
        shape_animation_lap = self.renderer_mr.render_yaw(canon_shape_im, self.canon_depth_lap, v_after=view_after, rotations=yaw_rotations)  # BxTxCxHxW
        self.shape_animation_lap = rearrange_frames(shape_animation_lap)

        ## normal map
        canon_normal_im = self.canon_normal_lap.permute(0,3,1,2) /2+0.5
        normal_animation = self.renderer_mr.render_yaw(canon_normal_im, self.canon_depth_lap, v_after=view_after, rotations=yaw_rotations)  # BxTxCxHxW
        self.normal_animation_lap = rearrange_frames(normal_animation)

        ## textured
        texture_animation = self.renderer_mr.render_yaw(self.canon_im_lap /2+0.5, self.canon_depth_lap, v_after=view_after, rotations=yaw_rotations)  # BxTxCxHxW
        self.texture_animation_lap = rearrange_frames(texture_animation)

    def save_results(self, save_dir):
        print(f"Saving results to {save_dir}")
        save_image(save_dir, self.input_im_lap[0]/2+0.5, 'input_image')
        save_image(save_dir, self.depth_inv_rescaler(self.canon_depth_lap)[0].repeat(3,1,1), 'canonical_depth_lap')
        save_image(save_dir, self.canon_normal_lap[0].permute(2,0,1)/2+0.5, 'canonical_normal_lap')
        save_image(save_dir, self.depth_inv_rescaler(self.recon_depth)[0].repeat(3,1,1), 'recon_depth_lap')
        save_image(save_dir, self.recon_normal[0].permute(2,0,1)/2+0.5, 'recon_normal_lap')
        save_image(save_dir, self.canon_diffuse_shading_lap[0].repeat(3,1,1), 'canonical_diffuse_shading_lap')
        save_image(save_dir, self.canon_albedo_lap[0]/2+0.5, 'canonical_albedo_lap')
        save_image(save_dir, self.canon_im_lap[0].clamp(-1,1)/2+0.5, 'canonical_image_lap')
        
        with open(os.path.join(save_dir, 'result_lap.mtl'), "w") as f:
            f.write(self.mtls_lap[0].replace('$TXTFILE', './canonical_image_lap.png'))
        with open(os.path.join(save_dir, 'result_lap.obj'), "w") as f:
            f.write(self.objs_lap[0].replace('$MTLFILE', './result_lap.mtl'))

        if self.render_video:
            save_video(save_dir, self.shape_animation_lap[0], 'shape_animation_lap')
            save_video(save_dir, self.normal_animation_lap[0], 'normal_animation_lap')
            save_video(save_dir, self.texture_animation_lap[0], 'texture_animation_lap')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Demo configurations.')
    parser.add_argument('--input', default='./images', type=str, help='Path to the directory containing input images')
    parser.add_argument('--result', default='./results5', type=str, help='Path to the directory for saving results')
    parser.add_argument('--checkpoint_lap', default='./demo/checkpoint300.pth', type=str, help='Path to the checkpoint file')
    parser.add_argument('--output_size', default=128, type=int, help='Output image size')
    parser.add_argument('--gpu', default=True, action='store_true', help='Enable GPU')
    parser.add_argument('--detect_human_face', default=False, action='store_true', help='Enable automatic human face detection. This does not detect cat faces.')
    parser.add_argument('--render_video', default=False, action='store_true', help='Render 3D animations to video')
    args = parser.parse_args()

    input_dir = args.input
    result_dir = args.result
    model = Demo(args)
    im_list = [os.path.join(input_dir, f) for f in sorted(os.listdir(input_dir)) if is_image_file(f)]

    for im_path in im_list:
        print(f"Processing {im_path}")
        pil_im = Image.open(im_path).convert('RGB')
        result_code = model.run(pil_im)
        if result_code == -1:
            print(f"Failed! Skipping {im_path}")
            continue

        save_dir = os.path.join(result_dir, os.path.splitext(os.path.basename(im_path))[0])
        model.save_results(save_dir)
