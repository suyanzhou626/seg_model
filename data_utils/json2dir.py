import os
import numpy as np
import json
from PIL import Image as img
from PIL import ImageDraw as imgd
from labelme import utils

def _bit_get(val, idx):
    return (val >> idx) & 1
def _create_256_label_colormap():
    colormap = np.zeros((256, 3), dtype=int)
    ind = np.arange(256, dtype=int)
    for shift in reversed(range(8)):
        for channel in range(3):
            colormap[:, channel] |= _bit_get(ind, channel) << shift
        ind >>= 3
    return colormap


def drawPolygonWithBorder(draw, points, fill, border):
    points.append(points[0])
    draw.polygon(points, fill=fill)
    if os.getenv('DRAW_BORDER') == 'TRUE':
        width = 5
        hf = 2
        draw.line(points, fill=border, width=width)
        for point in points:
            draw.ellipse((point[0] - hf, point[1] - hf, point[0]  + hf, point[1] + hf), fill=border)

def json2dir(dir_json,dir_out,mode,config_path):
    print('process json to label of ',mode)
    colormap = _create_256_label_colormap()
    config_json = json.load(open(config_path,'r'))
    labels = None
    border_value = None
    for model in config_json['models_to_labels']:
        if mode == model['name']:
            labels = model['labels']
            border_value = model['border_value']
            break
    if labels == None or border_value == None:
        print('this config dont include this mode!')
        raise EOFError

    dirname = dir_json
    fns = [x for x in os.listdir(dirname) if x.endswith('.json')]
    print('num of file: %d' % len(fns))

    outdir = dir_out
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    if os.getenv('DRAW_BORDER') != 'TRUE':
        border_value -= 1
    fi = open(os.path.join(outdir,'num_classes.txt'),'w')
    fi.writelines(str(border_value+1))
    fi.close()

    save_image_dir = os.path.join(outdir, 'image')
    if not os.path.exists(save_image_dir):
        os.mkdir(save_image_dir)

    save_labelclass_dir = os.path.join(outdir, 'labelclass')
    if not os.path.exists(save_labelclass_dir):
        os.mkdir(save_labelclass_dir)

    save_labelraw_dir = os.path.join(outdir, 'labelraw')
    if not os.path.exists(save_labelraw_dir):
        os.mkdir(save_labelraw_dir)

    count = 1
    discard = 0
    for i in fns:
        if count % 100 == 0:
            print('     generate the label, %d/%d' % (count,len(fns)))
        
        json_file = os.path.join(dirname,i)
        try:
            data = json.load(open(json_file))
        except:
            discard += 1
            continue
        try:
            image = utils.img_b64_to_arr(data['imageData'])
        except:
            discard += 1
            continue
        image = img.fromarray(image.astype(np.uint8),mode="RGB")
        img_shape = (image.size[1],image.size[0])
        scale = 600.00/float(max(image.size[1],image.size[0]))
        assert(type(scale) == float)
        target_size = (int(float(image.size[0])*scale),int(float(image.size[1])*scale))
        image = image.resize(target_size)
        image_name_base = '%d' % count
        save_image_name = os.path.join(save_image_dir, '%s.png' % image_name_base)

        save_mask_name = os.path.join(save_labelraw_dir,'%s.png' % image_name_base)
        save_color_name = os.path.join(save_labelclass_dir,'%s.png' % image_name_base)
        mask = np.zeros(img_shape,dtype=np.uint8)
        mask = img.fromarray(mask)
        draw = imgd.Draw(mask)
        for shape in data['shapes']: 
            if 'd' == mode[0]:
                temp_shape = shape['label'][0:3]
            else:
                temp_shape = shape['label'][0:4]
            for mask_value,label_index in labels.items():
                if temp_shape in label_index:
                    xy = list(map(tuple,shape['points']))
                    drawPolygonWithBorder(draw, xy, int(mask_value), border_value)
                    break
                    # draw.polygon(xy,fill = mask_value)
                elif shape['label'] == '255ignore':
                    xy = list(map(tuple,shape['points']))     
                    draw.polygon(xy,fill = 255)
                    break      
        mask_array = np.array(mask)
        assert(1<=len(np.unique(mask_array))<=border_value+1)
        array_max = np.max(mask_array)
        if 1<= array_max <= border_value+1 or (array_max==255 and len(np.unique(mask_array)) >= 3):
            color_mask = img.fromarray(np.uint8(colormap[np.array(mask,dtype=np.uint8)]))
            image.save(save_image_name)
            mask = mask.resize(target_size)
            mask.save(save_mask_name)
            color_mask = color_mask.resize(target_size)
            color_mask.save(save_color_name)
            
        else:
#            print('     this image have no damage of %s: ' % mode,i)
            discard += 1
        count += 1
    print('the num having no damage of %s is: '% mode,discard)
