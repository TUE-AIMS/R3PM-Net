import time
import copy
import os
import torch
from thirdparty.learning3d.models import PointNet, PointNetLK, DCP, DGCNN, iPCRNet, PRNet, PPFNet, RPMNet
from r3pm_net.model import R3PMNet
from r3pm_net.feature_extractor import feature_extractor # import your feature extractor here

from tools import metrics, l3d_helper

def l3d_reg_and_eval(source, target, method, gt_transformation, args, source_normals = None, target_normals = None):
    '''
    Perform registration and evaluation for a given Lerning3d method.

    Args:
        source (o3d.geometry.PointCloud): source point cloud
        target (o3d.geometry.PointCloud): target point cloud
        method (str): method name (e.g., 'dcp', 'rpmnet', 'pcrnet', 'pointnetlk')
        gt_transformation (np.ndarray): ground truth transformation matrix
        args (argparse.Namespace): arguments
    
    Returns:
        evaluation_results (np.ndarray): The evaluation results (rmse, rotation_error, translation_error, computation_time).
    '''

    # define the model
    if method == 'dcp':
        dgcnn = DGCNN(emb_dims=args.emb_dims)
        model = DCP(feature_model=dgcnn, cycle=True)
    elif method == 'rpmnet':
        model = RPMNet(feature_model=PPFNet())
    elif method == 'pcrnet':
        ptnet = PointNet(emb_dims=args.emb_dims)
        model = iPCRNet(feature_model=ptnet)
    elif method == 'pointnetlk':
        ptnet = PointNet(emb_dims=args.emb_dims, use_bn=True)
        model = PointNetLK(feature_model=ptnet)
    elif method == 'prnet':
        model = PRNet(emb_dims=args.emb_dims, num_iters=args.num_iterations)
    elif method == 'r3pmnet':
        FEATURE_MODEL = feature_extractor
        model = R3PMNet(feature_model=FEATURE_MODEL)
    else:
        raise ValueError(f"Unknown method: {method}")

    # load pretrained model
    if args.pretrained:
        if not os.path.isfile(args.pretrained):
            raise FileNotFoundError(f"Pretrained checkpoint not found.")
        try:
            payload = torch.load(args.pretrained, weights_only=False)
        except TypeError:
            payload = torch.load(args.pretrained)
        model.load_state_dict(payload, strict=False)

    # move model to device and set to eval mode
    model = model.to(args.device)
    model.eval()
    
    if source_normals is not None and target_normals is not None:
        source_tensor = l3d_helper.add_normal(source, source_normals, args.device)
        target_tensor = l3d_helper.add_normal(target, target_normals, args.device)
        print(source_tensor.shape)

    else:
        # convert data to tensor
        source_tensor = l3d_helper.convert_data(source, args.device)
        target_tensor = l3d_helper.convert_data(target, args.device)

    # Perform warm-up run (to avoid slow first run)
    model(target_tensor, source_tensor)

    # perform registration
    start_time = time.time()
    output = model(target_tensor, source_tensor)
    end_time = time.time()
    computation_time = end_time - start_time

    # Apply transformation
    result = output['est_T'].detach().cpu().numpy()[0]
    result = result.reshape(4, 4)
    pc_result = copy.deepcopy(source).transform(result)

    # Evaluation
    evaluation_results = metrics.all_evaluations(source, target, pc_result, computation_time, gt_transformation, output, corres = None)

    return pc_result, evaluation_results