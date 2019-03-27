import os
import numpy as np
import json
from PIL import Image as img
from PIL import ImageDraw as imgd

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

def json2dir(dir_json,dir_out,mode):
    print('process json to label of ',mode)
    colormap = _create_256_label_colormap()
    if mode == 'rightbehind':
        labels = {1:['p100'], 2:['p080'], 3:['p094'], 4:['p092'], 5:['p096'], 
                    6:['p085'], 7:['p004'],8:['p102'],9:['p007'],10:['p002'],11:['p016'],12:['p106'],13:['p083'],14:['p087'],
                    15:['p088'],16:['p101'],17:['p065']}
        border_value = 18
    elif mode == 'leftbehind':
        labels = {1:['p099'], 2:['p080'], 3:['p093'], 4:['p091'], 5:['p095'], 
                    6:['p085'], 7:['p003'], 8:['p102'],9:['p005'],10:['p001'],11:['p015'],12:['p105'],13:['p083'],14:['p087'],
                    15:['p088'],16:['p101'],17:['p064']}
        border_value = 18
    elif mode == 'rightfront':
        labels = {1:['p050'], 2:['p039'], 3:['p043'], 4:['p037'], 5:['p035'], 6:['p102'], 
                    7:['p020'], 8:['p034'], 9:['p031'], 10:['p029'],11:['p023'],12:['p018'],13:['p014'],14:['p032'],15:['p036'],
                    16:['p041'],17:['p065'],18:['p101'],19:['p075','p077','p079']}
        border_value = 20
    elif mode == 'leftfront':
        labels = {1:['p050'], 2:['p038'], 3:['p042'], 4:['p037'], 5:['p035'], 6:['p102'], 
                    7:['p019'], 8:['p034'], 9:['p030'], 10:['p029'],11:['p021'],12:['p017'],13:['p013'],14:['p032'],15:['p036'],
                    16:['p040'],17:['p064'],18:['p101'],19:['p074','p076','p078']}
        border_value = 20
    elif mode == 'behind':
        labels = {1:['p080'], 2:['p085'],3:['p083'],4:['p087'],5:['p088'],6:['p091'],7:['p092'],8:['p093'],9:['p094'],10:['p099'],
                11:['p100']}
        border_value = 12
    elif mode == 'front':
        labels = {1:['p034'],2:['p035'],3:['p038','p039'],4:['p037'],5:['p050']}
        border_value = 6
    elif mode == 'right':
        labels = {1:['p034'],2:['p039'],3:['p037'],4:['p043'],5:['p101', 'p102'],6:['p020'],7:['p014'],8:['p004'],9:['p106','p016'],
                    10:['p096'],11:['p080'],12:['p092']}
        border_value = 13
    elif mode == 'left':
        labels = {1:['p034'],2:['p038'],3:['p037'],4:['p042'],5:['p101', 'p102'],6:['p019'],7:['p013'],8:['p003'],9:['p105','p015'],
                    10:['p095'],11:['p080'],12:['p091']}
        border_value = 13
    elif mode == 'd01':
        labels = {1:['d01','d03']}
        border_value = 2 
    elif mode == 'd02':
        labels = {1:['d02']}
        border_value = 2
    elif mode == 'd05':
        labels = {1:['d05','d06']}
        border_value = 2
    else:
        labels = {}
        border_value = 2

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
        image_file = json_file.replace('.json','_json')+'/img.png'
        if not os.path.exists(json_file.replace('.json','_json')):
            try:
                os.system('labelme_json_to_dataset %s' % json_file)
            except:
                exit
        if not os.path.exists(image_file):
            # print('     the {} dont has relevant picture file'.format(json_file))
            discard += 1
            continue
            
        image = img.open(image_file)
        data = json.load(open(json_file))
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
                    drawPolygonWithBorder(draw, xy, mask_value, border_value)
                    # draw.polygon(xy,fill = mask_value)
                elif shape['label'] == '255ignore':
                    xy = list(map(tuple,shape['points']))     
                    draw.polygon(xy,fill = 255)      
        mask_array = np.array(mask)
        assert(1<=len(np.unique(mask_array))<=25)
        array_max = np.max(mask_array)
        if 1<= array_max <= 25 or (array_max==255 and len(np.unique(mask_array)) >= 3):
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
