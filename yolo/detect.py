from __future__ import division
from yolo.util import *

classes = load_classes("yolo/data/coco.names")

def write(x, results, bbc):
    c1 = tuple(x[0][1:3].int())
    c2 = tuple(x[0][3:5].int())
    
    # controlla che la persona detectata non sia contenuta dentro una bounding boxes di un quadro
    for c in bbc:
        if c1[0].item() >= c[0] and c1[1].item() >= c[1] and c2[0].item() <= c[0] + c[2] and c2[1].item() <= c[1] + c[3] - 10:
            return None

    # se detecto delle cose bianche tipo statue, condizionatori, ecc... non li considero persone
    person = results[c1[1]:c2[1], c1[0]:c2[0]]
    hist_b = cv2.calcHist([person], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([person], [1], None, [256], [0, 256])
    hist_r = cv2.calcHist([person], [2], None, [256], [0, 256])
    hist= (hist_b+hist_g+hist_r) /3
    if np.argmax(hist) > 50:
        return None

    img = results
    cv2.rectangle(img, c1, c2, (230, 65, 203), 1)
    t_size = cv2.getTextSize('Person', cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 8
    cv2.rectangle(img, c1, c2, (230, 65, 203), -1)
    cv2.putText(img, 'Person', (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225, 255, 255], 1);


def people_detection(im, bbc, model):
    batch_size = 1
    confidence = 0.7
    nms_thesh = 0.4

    CUDA = torch.cuda.is_available()

    num_classes = 80

    model.net_info["height"] = 416
    inp_dim = int(model.net_info["height"])
    assert inp_dim % 32 == 0
    assert inp_dim > 32

    # If there's a GPU availible, put the model on GPU
    if CUDA:
        model.cuda()

    # Set the model in evaluation mode
    model.eval()

    loaded_ims = [im]

    im_batches = list(map(prep_image, loaded_ims, [inp_dim for x in range(len(loaded_ims))]))
    im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]
    im_dim_list = torch.FloatTensor(im_dim_list).repeat(1, 2)

    leftover = 0
    if (len(im_dim_list) % batch_size):
        leftover = 1

    if batch_size != 1:
        num_batches = 1 // batch_size + leftover
        im_batches = [torch.cat((im_batches[i * batch_size: min((i + 1) * batch_size,
                                                                len(im_batches))])) for i in range(num_batches)]

    write_int = 0

    if CUDA:
        im_dim_list = im_dim_list.cuda()

    for i, batch in enumerate(im_batches):
        # load the image
        if CUDA:
            batch = batch.cuda()
        with torch.no_grad():
            prediction = model(Variable(batch), CUDA)

        prediction = write_results(prediction, confidence, num_classes, nms_conf=nms_thesh)

        if type(prediction) == int:
            continue

        prediction[:, 0] += i * batch_size  # transform the atribute from index in batch to index in imlist

        if not write_int:  # If we have't initialised output
            output = prediction
            write_int = 1
        else:
            output = torch.cat((output, prediction))

        if CUDA:
            torch.cuda.synchronize()
    try:
        output
    except NameError:
        return None

    im_dim_list = torch.index_select(im_dim_list, 0, output[:, 0].long())

    scaling_factor = torch.min(416 / im_dim_list, 1)[0].view(-1, 1)

    output[:, [1, 3]] -= (inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
    output[:, [2, 4]] -= (inp_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2

    output[:, 1:5] /= scaling_factor

    for i in range(output.shape[0]):
        output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0.0, im_dim_list[i, 0])
        output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0.0, im_dim_list[i, 1])

    write(output, im, bbc)

    torch.cuda.empty_cache()

