import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn.config import Config
from mrcnn import model as modellib, utils
# from mrcnn import visualize as vzl
from mrcnn.visualize import display_images
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")


class IdentityConfig(Config):
    """Configuration for training on the toy  dataset.
        Derives from the base Config class and overrides some values.
        """
    # Give the configuration a recognizable name
    NAME = "identity"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1  # Background + identity

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 90% confidence
    DETECTION_MIN_CONFIDENCE = 0.9


    RPN_ANCHOR_RATIOS = [0.75, 1.5, 3]
    RPN_BBOX_STD_DEV = np.array([0.22, 0.22, 0.22, 0.22])
    RPN_ANCHOR_SCALES = (64, 128, 256, 512, 1024)
    DETECTION_NMS_THRESHOLD = 0.9



class IdentityDataset(utils.Dataset):
    def load_identity(self, dataset_dir, subset):
        # Add classes
        self.add_class("identity", 1, "identity")

        # Train or validation dataset
        assert subset in ["train", "val"]
        dataset_dir = os.path.join(dataset_dir, subset)

        annotations = json.load(open(os.path.join(dataset_dir, "via_region_data.json")))
        annotations = list(annotations.values())  # don't need the dict keys

        # The VIA tool saves images in the JSON even if they don't have any
        # annotations. Skip unannotated images.
        annotations = [a for a in annotations if a['regions']]

        # Add images
        for a in annotations:
            # Get the x, y coordinaets of points of the polygons that make up
            # the outline of each object instance. There are stores in the
            # shape_attributes (see json format above)
            polygons = [r['shape_attributes'] for r in a['regions'].values()]

            # load_mask() needs the image size to convert polygons to masks.
            # Unfortunately, VIA doesn't include it in JSON, so we must read
            # the image. This is only managable since the dataset is tiny.
            image_path = os.path.join(dataset_dir, a['filename'])
            image = skimage.io.imread(image_path)
            height, width = image.shape[:2]

            self.add_image(
                "identity",
                image_id=a['filename'],  # use file name as a unique image id
                path=image_path,
                width=width, height=height,
                polygons=polygons)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a identity dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "identity":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])],
                        dtype=np.uint8)
        for i, p in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "identity":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


class_names = ['BG', 'identity']


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = IdentityDataset()
    dataset_train.load_identity(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = IdentityDataset()
    dataset_val.load_identity(args.dataset, "val")
    dataset_val.prepare()

    # *** This training schedule is an example. Update to your needs ***
    # Since we're using a very small dataset, and starting from
    # COCO trained weights, we don't need to train too long. Also,
    # no need to train all layers, just the heads should do it.
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=1,
                layers='heads')


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 100
    plt.imshow(gray)
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


# def final_detection(model, image_path=None, video_path=None):
#     assert image_path or video_path
#
#     # Image or video?
#     if image_path:
#         image = skimage.io.imread(args.image)
#         # Get input and output to classifier and mask heads.
#         mrcnn = model.run_graph([image], [
#             ("proposals", model.keras_model.get_layer("ROI").output),
#             ("probs", model.keras_model.get_layer("mrcnn_class").output),
#             ("deltas", model.keras_model.get_layer("mrcnn_bbox").output),
#             ("masks", model.keras_model.get_layer("mrcnn_mask").output),
#             ("detections", model.keras_model.get_layer("mrcnn_detection").output),
#         ])
#         # Get detection class IDs. Trim zero padding.
#         det_class_ids = mrcnn['detections'][0, :, 4].astype(np.int32)
#         det_count = np.where(det_class_ids == 0)[0][0]
#         det_class_ids = det_class_ids[:det_count]
#         detections = mrcnn['detections'][0, :det_count]
#
#         captions = ["{} {:.3f}".format(class_names[int(c)], s) if c > 0 else ""
#                     for c, s in zip(detections[:, 4], detections[:, 5])]
#         visualize.draw_boxes(
#             image,
#             refined_boxes=utils.denorm_boxes(detections[:, :4], image.shape[:2]),
#             visibilities=[2] * len(detections),
#             captions=captions, title="Detections",
#             ax=get_ax())
#         # Get detection class IDs. Trim zero padding.
#         det_class_ids = mrcnn['detections'][0, :, 4].astype(np.int32)
#         det_count = np.where(det_class_ids == 0)[0][0]
#         det_class_ids = det_class_ids[:det_count]
#         detections = mrcnn['detections'][0, :det_count]
#         # Proposals are in normalized coordinates. Scale them
#         # to image coordinates.
#         h, w = config.IMAGE_SHAPE[:2]
#         proposals = np.around(mrcnn["proposals"][0] * np.array([h, w, h, w])).astype(np.int32)
#         # Class ID, score, and mask per proposal
#         roi_class_ids = np.argmax(mrcnn["probs"][0], axis=1)
#         roi_scores = mrcnn["probs"][0, np.arange(roi_class_ids.shape[0]), roi_class_ids]
#         roi_class_names = np.array(class_names)[roi_class_ids]
#         roi_positive_ixs = np.where(roi_class_ids > 0)[0]
#         # Class-specific bounding box shifts.
#         roi_bbox_specific = mrcnn["deltas"][0, np.arange(proposals.shape[0]), roi_class_ids]
#         # Apply bounding box transformations
#         # Shape: [N, (y1, x1, y2, x2)]
#         refined_proposals = utils.apply_box_deltas(
#             proposals, roi_bbox_specific * config.BBOX_STD_DEV).astype(np.int32)
#         # Remove boxes classified as background
#         keep = np.where(roi_class_ids > 0)[0]
#         keep = np.intersect1d(keep, np.where(roi_scores >= config.DETECTION_MIN_CONFIDENCE)[0])
#         # Apply per-class non-max suppression
#         pre_nms_boxes = refined_proposals[keep]
#         pre_nms_scores = roi_scores[keep]
#         pre_nms_class_ids = roi_class_ids[keep]
#
#         nms_keep = []
#         for class_id in np.unique(pre_nms_class_ids):
#             # Pick detections of this class
#             ixs = np.where(pre_nms_class_ids == class_id)[0]
#             # Apply NMS
#             class_keep = utils.non_max_suppression(pre_nms_boxes[ixs],
#                                             pre_nms_scores[ixs],
#                                             config.DETECTION_NMS_THRESHOLD)
#             # Map indicies
#             class_keep = keep[ixs[class_keep]]
#             nms_keep = np.union1d(nms_keep, class_keep)
#         keep = np.intersect1d(keep, nms_keep).astype(np.int32)
#         # Show final detections
#         ixs = np.arange(len(keep))  # Display all
#         # ixs = np.random.randint(0, len(keep), 10)  # Display random sample
#         captions = ["{} {:.3f}".format(class_names[c], s) if c > 0 else ""
#                     for c, s in zip(roi_class_ids[keep][ixs], roi_scores[keep][ixs])]
#         visualize.draw_boxes(
#                 image, boxes=proposals[keep][ixs],
#                 refined_boxes=refined_proposals[keep][ixs],
#                 visibilities=np.where(roi_class_ids[keep][ixs] > 0, 1, 0),
#                 captions=captions, title="Detections after NMS",
#                 ax=get_ax())


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave('result/' + file_name, splash)
        # vzl.display_instances(image, r['rois'], r['masks'], r['class_ids'],
        #                             ['BG', 'identity'], r['scores'],
        #                             title="Predictions")
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to detect identity.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/identity/dataset/",
                        help='Directory of the Identity dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=logs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video, \
            "Provide --image or --video to apply color splash"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = IdentityConfig()
    else:
        class InferenceConfig(IdentityConfig):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1


        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config,
                                  model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config,
                                  model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching
        # number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc",
            "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image,
                                video_path=args.video)
        # final_detection(model, image_path=args.image, video_path=args.video)
    else:
        print("'{}' is not recognized. "
              "Use 'train' or 'splash'".format(args.command))


