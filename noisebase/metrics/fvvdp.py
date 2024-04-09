from pyfvvdp.fvvdp import fvvdp, fvvdp_contrast_pyr, fvvdp_lpyr_dec, reshuffle_dims, fvvdp_video_source_dm
import torch
import numpy as np

class fvvdp_feeder:
    """fvvdp.predict_video_source and video_source.fvvdp_video_source_array hacked into one convenient package"""

    def __init__(self, parent_fvvdp, height, width, N_frames):
        self.fvvdp = parent_fvvdp
        self.height = height
        self.width = width
        self.N_frames = N_frames
        
        self.sw_buf = [None, None]
        self.Q_per_ch = None
        self.ff = 0
        self.dm = fvvdp_video_source_dm(parent_fvvdp.display_photometry, parent_fvvdp.color_space)

    def feed(self, cur_tframe, cur_rframe):
        """Feed frames to FovVideoVDP

        Args:
            cur_tframe (tensor, N3HW): frame to be tested
            cur_rframe (tensor, N3HW): reference frame
        """

        def to_fvvdp(nchw):
            frame_t = reshuffle_dims(nchw, in_dims='BCHW', out_dims="BCFHW")
            frame_t = frame_t.to(self.fvvdp.device)

            L = self.dm.dm_photometry.forward(frame_t)
            L = L[:,0:1,:,:,:]*self.dm.color_to_luminance[0] + L[:,1:2,:,:,:]*self.dm.color_to_luminance[1] + L[:,2:3,:,:,:]*self.dm.color_to_luminance[2]
            return L
        
        cur_tframe = to_fvvdp(cur_tframe)
        cur_rframe = to_fvvdp(cur_rframe)

        temp_ch = 2

        if self.ff == 0: # First frame. temp_padding="replicate"
            self.sw_buf[0] = cur_tframe.expand([1, 1, self.fvvdp.filter_len, self.height, self.width])
            self.sw_buf[1] = cur_rframe.expand([1, 1, self.fvvdp.filter_len, self.height, self.width])
        else:
            self.sw_buf[0] = torch.cat((self.sw_buf[0][:, :, 1:, :, :], cur_tframe), 2)
            self.sw_buf[1] = torch.cat((self.sw_buf[1][:, :, 1:, :, :], cur_rframe), 2)
        
        # Order: test-sustained, ref-sustained, test-transient, ref-transient
        R = torch.zeros((1, 4, 1, self.height, self.width), device=self.fvvdp.device)

        for cc in range(temp_ch):
            # 1D filter over time (over frames)
            corr_filter = self.fvvdp.F[cc].flip(0).view([1,1,self.fvvdp.F[cc].shape[0],1,1]) 
            R[:,cc*2+0, :, :, :] = (self.sw_buf[0] * corr_filter).sum(dim=-3,keepdim=True)
            R[:,cc*2+1, :, :, :] = (self.sw_buf[1] * corr_filter).sum(dim=-3,keepdim=True)

        Q_per_ch_block = self.fvvdp.process_block_of_frames(self.ff, R, (self.height, self.width, self.N_frames), temp_ch, torch.tensor([self.width//2, self.height//2]), None)

        if self.Q_per_ch is None:
            self.Q_per_ch = torch.zeros((Q_per_ch_block.shape[0], Q_per_ch_block.shape[1], self.N_frames), device=self.fvvdp.device)
        
        self.Q_per_ch[:,:,self.ff:(self.ff+Q_per_ch_block.shape[2])] = Q_per_ch_block
        self.ff += 1
    
    def compute(self):
        """Compute FovVideoVDP metric

        Returns:
            Q_jod (float): JOD scale FovVideoVDP value
        """
        rho_band = self.fvvdp.lpyr.get_freqs()
        Q_jod = self.fvvdp.do_pooling_and_jods(self.Q_per_ch, rho_band[0:-1])

        return float(Q_jod.squeeze())

class nb_fvvdp(fvvdp):
    """Proxy class to add video feeder"""

    def __init__(self, display_name="standard_4k", display_photometry=None, display_geometry=None, color_space="sRGB"):
        # Some options simplified
        super().__init__(
            display_name=display_name,
            display_photometry=display_photometry,
            display_geometry=display_geometry,
            color_space=color_space,
            foveated=False,
            heatmap=None,
            quiet=False, # TRUE
            temp_padding="replicate",
            use_checkpoints=False
        )

    def video_feeder(self, height, width, N_frames, fps):
        """Make a convenient video feeder for easy loading of videos

        Args:
            height (int): height of the video in pixels
            width (int): width of the video in pixels
            N_frames (int): number of frames in the video
            fps (float): video frames per second
        
        Returns:
            feeder (fvvdp_feeder): feed frames to this object using
            feeder.feed and get the metric using feeder.compute
        """
        # Some setup from predict_video_source
        if self.lpyr is None or self.lpyr.W!=width or self.lpyr.H!=height:
            if self.local_adapt=="gpyr":
                self.lpyr = fvvdp_contrast_pyr(width, height, self.pix_per_deg, self.device)
            else:
                self.lpyr = fvvdp_lpyr_dec(width, height, self.pix_per_deg, self.device)

        self.filter_len = int(np.ceil( 250.0 / (1000.0/fps) ))
        self.F, self.omega = self.get_temporal_filters(fps)

        return fvvdp_feeder(self, height, width, N_frames)