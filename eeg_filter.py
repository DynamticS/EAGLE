import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EEGBandFilter(nn.Module):
    def __init__(self, num_channels=32, signal_length=512, fs=128, numtaps=101):
        super(EEGBandFilter, self).__init__()
        
        self.fs = fs
        self.numtaps = numtaps
        self.num_channels = num_channels
        self.signal_length = signal_length
        
        self.bands = {
            'delta': (0.5, 4),
            'theta': (4, 7), 
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }
        
        self.register_buffer('filter_kernels', self._create_all_band_filters())
        
    def _create_all_band_filters(self):
        nyquist = 0.5 * self.fs
        kernels = []
        
        for band_name, (low, high) in self.bands.items():
            low_norm = low / nyquist
            high_norm = high / nyquist
            
            n = torch.arange(self.numtaps) - (self.numtaps - 1) / 2
            
            with torch.no_grad():
                kernel = (torch.sin(np.pi * n * high_norm) - torch.sin(np.pi * n * low_norm)) / (np.pi * n)
                kernel[(self.numtaps - 1) // 2] = high_norm - low_norm
                
                window = torch.hann_window(self.numtaps)
                kernel *= window
                
                kernels.append(kernel)
        
        return torch.stack(kernels)  # [5, numtaps]
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        if x.dim() == 2:
            x = x.view(batch_size, self.num_channels, self.signal_length)
        
        # x: [batch_size, num_channels, signal_length]
        # filter_kernels: [5, numtaps]
        filtered_signals = self._apply_band_filters(x)
        
        de_features = self._compute_differential_entropy(filtered_signals)
        
        return de_features
    
    def _apply_band_filters(self, x):
        batch_size, num_channels, signal_length = x.shape
        
        # x: [batch_size * num_channels, 1, signal_length]
        x_reshaped = x.view(batch_size * num_channels, 1, signal_length)
        
        # filter_kernels: [5, numtaps] -> [5, 1, numtaps]
        kernels = self.filter_kernels.unsqueeze(1)
        
        # kernels: [5 * num_channels, 1, numtaps]
        kernels_expanded = kernels.repeat(num_channels, 1, 1)
        
        filtered = F.conv1d(
            x_reshaped.repeat(1, 5, 1),  # [batch_size * num_channels, 5, signal_length]
            kernels_expanded,  # [5 * num_channels, 1, numtaps]
            groups=batch_size * num_channels,
            padding='same'
        )
        
        # filtered: [batch_size, num_channels, 5, signal_length]
        filtered = filtered.view(batch_size, num_channels, 5, signal_length)
        
        return filtered
    
    def _compute_differential_entropy(self, filtered_signals):
        # variance: [batch_size, num_channels, 5]
        variance = torch.var(filtered_signals, dim=-1, unbiased=False)
        
        epsilon = 1e-9
        de_features = 0.5 * torch.log(2 * np.pi * np.e * (variance + epsilon))
        
        return de_features
    
    def get_band_names(self):
        ""            
        return list(self.bands.keys())
    
    def get_band_ranges(self):
        ""          
        return self.bands


class EEGBandFilterOptimized(nn.Module):
    def __init__(self, num_channels=32, signal_length=512, fs=128, numtaps=101):
        super(EEGBandFilterOptimized, self).__init__()
        
        self.fs = fs
        self.numtaps = numtaps
        self.num_channels = num_channels
        self.signal_length = signal_length
        
        band_ranges = torch.tensor([
            [0.5, 4],    # delta
            [4, 7],      # theta  
            [8, 13],     # alpha
            [13, 30],    # beta
            [30, 45]     # gamma
        ])
        
        self.register_buffer('band_ranges', band_ranges)
        
        self.register_buffer('filter_kernels', self._create_vectorized_filters())
        
    def _create_vectorized_filters(self):
        nyquist = 0.5 * self.fs
        
        normalized_bands = self.band_ranges / nyquist
        lows = normalized_bands[:, 0]  # [5]
        highs = normalized_bands[:, 1]  # [5]
        
        n = torch.arange(self.numtaps, dtype=torch.float32) - (self.numtaps - 1) / 2
        
        n_expanded = n.unsqueeze(0)  # [1, numtaps]
        lows_expanded = lows.unsqueeze(1)  # [5, 1]
        highs_expanded = highs.unsqueeze(1)  # [5, 1]
        
        kernels = (torch.sin(np.pi * n_expanded * highs_expanded) - 
                  torch.sin(np.pi * n_expanded * lows_expanded)) / (np.pi * n_expanded)
        
        center_idx = (self.numtaps - 1) // 2
        kernels[:, center_idx] = highs - lows
        
        window = torch.hann_window(self.numtaps).unsqueeze(0)  # [1, numtaps]
        kernels *= window
        
        return kernels  # [5, numtaps]
    
    def filter_signal(self, x):
        """
        Apply 5 bandpass filters once over the full segment.
        Args:
            x: [batch_size, num_channels, signal_length]
        Returns:
            filtered: [batch_size, num_channels, 5, signal_length]
        """
        batch_size, num_channels, signal_length = x.shape
        x_flat = x.view(-1, 1, signal_length)
        kernels = self.filter_kernels.unsqueeze(1)  # [5, 1, numtaps]
        filtered_all = []
        for i in range(5):
            filtered_all.append(F.conv1d(x_flat, kernels[i:i + 1], padding='same'))
        filtered_stacked = torch.stack(filtered_all, dim=0)
        return filtered_stacked.permute(1, 2, 0, 3).squeeze(1).view(
            batch_size, num_channels, 5, signal_length
        )

    @staticmethod
    def de_from_filtered(filtered, epsilon=1e-6):
        """DE = 0.5 * log(2πe (var+eps)) over last dim. filtered: [..., L] → [...,]."""
        variance = torch.var(filtered, dim=-1, unbiased=False)
        return 0.5 * torch.log(2 * np.pi * np.e * (variance + epsilon))

    @staticmethod
    def de_patch_from_filtered(filtered, patch_size, num_patch, de_window, epsilon=1e-6):
        """
        Per-patch DE via overlapping windows centered on each patch (LE path only).

        filtered: [B, C, 5, L] with L = patch_size * num_patch.
        Returns de_patch: [B, C, 5, P] then caller may permute to [B, P, C, 5].

        Window length de_window must be long enough for delta (0.5Hz): at least
        ~0.5 period → de_window >= fs/(2*0.5); at fs=128 that is 128 samples.
        Patch positions stay at stride=patch_size (fine P); DE uses wider centered window.
        """
        B, C, n_bands, L = filtered.shape
        ps = int(patch_size)
        P = int(num_patch)
        W = int(de_window)
        if L != ps * P:
            raise ValueError(f'expected L=patch_size*num_patch={ps*P}, got L={L}')
        if W < ps:
            raise ValueError(f'de_window={W} must be >= patch_size={ps}')
        if W % 2 != 0:
            raise ValueError(f'de_window must be even for symmetric centering, got {W}')
        # Align window centers with patch centers: left_pad = W/2 - ps/2
        left_pad = W // 2 - ps // 2
        right_pad = (W - ps) - left_pad
        # reflect pad on time: reshape to 3D for broad PyTorch reflect support
        flat = filtered.reshape(B * C * n_bands, 1, L)
        flat_pad = torch.nn.functional.pad(flat, (left_pad, right_pad), mode='reflect')
        filt_pad = flat_pad.reshape(B, C, n_bands, L + left_pad + right_pad)
        # unfold → [B, C, 5, P, W]; step=ps yields exactly P windows
        windows = filt_pad.unfold(-1, W, ps)
        if windows.shape[-2] != P:
            raise RuntimeError(
                f'de_patch unfold P mismatch: got {windows.shape[-2]} expected {P}; '
                f'shape={tuple(windows.shape)} left_pad={left_pad} right_pad={right_pad} W={W}'
            )
        return EEGBandFilterOptimized.de_from_filtered(windows, epsilon=epsilon)  # [B,C,5,P]

    def forward(self, x, num_windows=1, return_windows=False):
        """
        Filter once over the full L-point segment, then optionally split into T windows
        for per-window DE (GE multi-window path). EAPatch always uses full-segment DE.

        Args:
            x: [batch_size, num_channels, signal_length]
            num_windows: T ∈ {1,2,4}. Filter still runs once on full L.
            return_windows: if True, also return multi-window DE as [B, T, 5, C]
                (band-as-node layout for BandGraphGE).
        Returns:
            de_features: [B, C, 5]  (full-segment DE, backward compatible)
            or (de_features, x_ge) when return_windows=True, with
                x_ge: [B, T, 5, C]
        """
        filtered = self.filter_signal(x)  # [B, C, 5, L]
        epsilon = 1e-6
        de_features = self.de_from_filtered(filtered, epsilon=epsilon)
        if torch.isnan(de_features).any():
            print("Warning: de_features contains NaN values")

        if not return_windows:
            return de_features

        B, C, F, L = filtered.shape
        T = int(num_windows)
        if T <= 1:
            # [B,C,5] → [B,1,5,C]
            x_ge = de_features.permute(0, 2, 1).unsqueeze(1).contiguous()
            return de_features, x_ge

        assert L % T == 0, f'signal_length {L} not divisible by num_windows {T}'
        # reshape [B,C,5,T,L//T] → var over last → [B,C,5,T] → [B,T,5,C]
        win = filtered.view(B, C, F, T, L // T)
        de_win = self.de_from_filtered(win, epsilon=epsilon)  # [B,C,5,T]
        if torch.isnan(de_win).any():
            print("Warning: multi-window de_features contains NaN values")
        x_ge = de_win.permute(0, 3, 2, 1).contiguous()  # [B,T,5,C]
        return de_features, x_ge
