import xml.etree.ElementTree as ET
import os
import json


CLASS_NAMES = (
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
)


coco = dict()
coco["images"] = []
coco["type"] = "instances"
coco["annotations"] = []
coco["categories"] = []

category_set = dict()
image_set = set()

image_id = 000000
annotation_id = 0


def addCatItem(name):
    category_item = dict()
    category_item["supercategory"] = "none"
    cls_id = CLASS_NAMES.index(name)
    category_item["id"] = cls_id
    category_item["name"] = name
    coco["categories"].append(category_item)
    category_set[name] = cls_id
    return cls_id


def addImgItem(file_name, size):
    image_id = file_name.split(".")[0]
    image_item = dict()
    image_item["id"] = image_id
    image_item["file_name"] = file_name
    image_item["width"] = size["width"]
    image_item["height"] = size["height"]
    coco["images"].append(image_item)
    image_set.add(file_name)
    return image_id


def addAnnoItem(image_id, category_id, bbox):
    global annotation_id
    annotation_item = dict()
    annotation_item["area"] = bbox[2] * bbox[3]
    annotation_item["iscrowd"] = 0
    annotation_item["ignore"] = 0
    annotation_item["image_id"] = image_id
    annotation_item["bbox"] = bbox
    annotation_item["category_id"] = category_id
    annotation_id += 1
    annotation_item["id"] = annotation_id
    coco["annotations"].append(annotation_item)


def parseXmlFiles(xml_path):
    for f in os.listdir(xml_path):
        if not f.endswith(".xml"):
            continue

        bndbox = dict()
        size = dict()
        current_image_id = None
        current_category_id = None
        file_name = None
        size["width"] = None
        size["height"] = None
        size["depth"] = None

        xml_file = os.path.join(xml_path, f)

        tree = ET.parse(xml_file)
        root = tree.getroot()
        if root.tag != "annotation":
            raise Exception("pascal voc xml root element should be annotation, rather than {}".format(root.tag))

        # elem is <folder>, <filename>, <size>, <object>
        for elem in root:
            current_parent = elem.tag
            current_sub = None
            object_name = None

            if elem.tag == "folder":
                continue

            if elem.tag == "filename":
                file_name = elem.text
                if file_name in category_set:
                    raise Exception("file_name duplicated")

            # add img item only after parse <size> tag
            elif current_image_id is None and file_name is not None and size["width"] is not None:
                if file_name not in image_set:
                    current_image_id = addImgItem(file_name, size)
                else:
                    raise Exception("duplicated image: {}".format(file_name))
            # subelem is <width>, <height>, <depth>, <name>, <bndbox>
            for subelem in elem:
                bndbox["xmin"] = None
                bndbox["xmax"] = None
                bndbox["ymin"] = None
                bndbox["ymax"] = None

                current_sub = subelem.tag
                if current_parent == "object" and subelem.tag == "name":
                    object_name = subelem.text
                    if object_name not in category_set:
                        current_category_id = addCatItem(object_name)
                    else:
                        current_category_id = category_set[object_name]

                elif current_parent == "size":
                    if size[subelem.tag] is not None:
                        raise Exception("xml structure broken at size tag.")
                    size[subelem.tag] = int(subelem.text)

                # option is <xmin>, <ymin>, <xmax>, <ymax>, when subelem is <bndbox>
                for option in subelem:
                    if current_sub == "bndbox":
                        if bndbox[option.tag] is not None:
                            raise Exception("xml structure corrupted at bndbox tag.")
                        bndbox[option.tag] = int(option.text)

                # only after parse the <object> tag
                if bndbox["xmin"] is not None:
                    if object_name is None:
                        raise Exception("xml structure broken at bndbox tag")
                    if current_image_id is None:
                        raise Exception("xml structure broken at bndbox tag")
                    if current_category_id is None:
                        raise Exception("xml structure broken at bndbox tag")
                    bbox = []
                    # x
                    bbox.append(bndbox["xmin"] - 1)
                    # y
                    bbox.append(bndbox["ymin"] - 1)
                    # w
                    bbox.append(bndbox["xmax"])
                    # h
                    bbox.append(bndbox["ymax"])
                    addAnnoItem(current_image_id, current_category_id, bbox)


if __name__ == "__main__":
    xml_path = "Annotations"
    json_file = "instances.json"
    parseXmlFiles(xml_path)
    json.dump(coco, open(json_file, "w"))
