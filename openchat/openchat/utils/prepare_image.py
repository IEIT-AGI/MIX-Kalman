import torchvision.transforms as T
from PIL import Image
import io
import torch
import cv2


def transform_image(image_bytes, target_size):
    trans = T.Compose(
        [T.Resize(255),
         T.CenterCrop(244),
         T.ToTensor(),
         T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    image = Image.open(io.BytesIO(image_bytes))
    if image.mode != 'RGB':
        image = image.convert('RGB')

    return trans(image)


def cv2Img_to_Image(input_img):
    cv2_img = input_img.copy()
    img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    return img


def detect_objects_on_single_image(model, transforms, cv2_img):
    # cv2_img is the original input, so we can get the height and
    # width information to scale the output boxes.
    img_input = cv2Img_to_Image(cv2_img)
    img_input, _ = transforms(img_input, target=None)
    img_input = img_input.to(model.device)

    with torch.no_grad():
        prediction = model(img_input)
        prediction = prediction[0].to(torch.device('cpu'))

    img_height = cv2_img.shape[0]
    img_width = cv2_img.shape[1]

    prediction = prediction.resize((img_width, img_height))
    boxes = prediction.bbox.tolist()
    classes = prediction.get_field('labels').tolist()
    scores = prediction.get_field('scores').tolist()
    feats = prediction.get_field('box_features')

    if 'attr_scores' in prediction.extra_fields:
        attr_scores = prediction.get_field('attr_scores')
        attr_labels = prediction.get_field('attr_labels')
        return [{
            'rect': box,
            'class': cls,
            'conf': score,
            'attr': attr[attr_conf > 0.01].tolist(),
            'attr_conf': attr_conf[attr_conf > 0.01].tolist(),
            'features': feat
        } for box, cls, score, attr, attr_conf, feat in zip(boxes, classes, scores, attr_labels, attr_scores, feats)]

    return [{'rect': box, 'class': cls, 'conf': score} for box, cls, score in zip(boxes, classes, scores)]


def extract_visual_feature_on_single_image(model, transforms, img_data):
    img_input = cv2Img_to_Image(img_data)
    img_input, _ = transforms(img_input, target=None)
    img_input = img_input.to(model.device)
    img_h, img_w, _ = img_data.shape

    with torch.no_grad():
        prediction = model(img_input)
        return prediction[0].resize((img_w, img_h))
