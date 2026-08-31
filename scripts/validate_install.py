import torch
import cumm
import spconv.pytorch
import torchsparse
import MinkowskiEngine

from gtsparse.sparse3d.geometric_template import GeometricTemplateSubMConv3d
from gtsparse.sparse3d.reference_ops import reference_subm_conv3d
from gtsparse.sparse3d.sparse_tensor import GTSparseSparseConvTensor


torch.manual_seed(0)
grid = torch.cartesian_prod(
    torch.arange(1), torch.arange(1, 7), torch.arange(1, 7), torch.arange(1, 7)
).to(torch.int32)
coords = grid[torch.randperm(grid.size(0))[:96]].contiguous().cuda()
channels = 64
features = torch.randn(96, channels, device="cuda")

for dtype, atol, rtol in (
    (torch.float32, 1e-3, 1e-3),
    (torch.float16, 2e-2, 2e-2),
):
    conv = GeometricTemplateSubMConv3d(channels, channels, 3, padding=1).cuda().to(dtype)
    sparse = GTSparseSparseConvTensor(features.to(dtype), coords, (8, 8, 8), 1)
    with torch.no_grad():
        actual = conv(sparse)
        weight = conv.weight.view(3, 3, 3, channels, channels).permute(4, 3, 0, 1, 2).contiguous()
        expected = reference_subm_conv3d(sparse, weight, None, padding=1)
    torch.testing.assert_close(actual.features, expected.features, atol=atol, rtol=rtol)
    error = (actual.features - expected.features).abs().max().item()
    print(dtype, f"max_error={error:.6g}")
