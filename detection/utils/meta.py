class COCOMeta:
    def __init__(self):
        self.meta = [
            {"color": [220, 20, 60], "id": 1, "name": "person"},
            {"color": [119, 11, 32], "id": 2, "name": "bicycle"},
            {"color": [0, 0, 142], "id": 3, "name": "car"},
            {"color": [0, 0, 230], "id": 4, "name": "motorcycle"},
            {"color": [106, 0, 228], "id": 5, "name": "airplane"},
            {"color": [0, 60, 100], "id": 6, "name": "bus"},
            {"color": [0, 80, 100], "id": 7, "name": "train"},
            {"color": [0, 0, 70], "id": 8, "name": "truck"},
            {"color": [0, 0, 192], "id": 9, "name": "boat"},
            {"color": [250, 170, 30], "id": 10, "name": "traffic light"},
            {"color": [100, 170, 30], "id": 11, "name": "fire hydrant"},  # 10
            {"color": [220, 220, 0], "id": 13, "name": "stop sign"},
            {"color": [175, 116, 175], "id": 14, "name": "parking meter"},
            {"color": [250, 0, 30], "id": 15, "name": "bench"},
            {"color": [165, 42, 42], "id": 16, "name": "bird"},
            {"color": [255, 77, 255], "id": 17, "name": "cat"},
            {"color": [0, 226, 252], "id": 18, "name": "dog"},
            {"color": [182, 182, 255], "id": 19, "name": "horse"},
            {"color": [0, 82, 0], "id": 20, "name": "sheep"},
            {"color": [120, 166, 157], "id": 21, "name": "cow"},
            {"color": [110, 76, 0], "id": 22, "name": "elephant"},  # 20
            {"color": [174, 57, 255], "id": 23, "name": "bear"},
            {"color": [199, 100, 0], "id": 24, "name": "zebra"},
            {"color": [72, 0, 118], "id": 25, "name": "giraffe"},
            {"color": [255, 179, 240], "id": 27, "name": "backpack"},
            {"color": [0, 125, 92], "id": 28, "name": "umbrella"},
            {"color": [209, 0, 151], "id": 31, "name": "handbag"},
            {"color": [188, 208, 182], "id": 32, "name": "tie"},
            {"color": [0, 220, 176], "id": 33, "name": "suitcase"},
            {"color": [255, 99, 164], "id": 34, "name": "frisbee"},
            {"color": [92, 0, 73], "id": 35, "name": "skis"},  # 30
            {"color": [133, 129, 255], "id": 36, "name": "snowboard"},
            {"color": [78, 180, 255], "id": 37, "name": "sports ball"},
            {"color": [0, 228, 0], "id": 38, "name": "kite"},
            {"color": [174, 255, 243], "id": 39, "name": "baseball bat"},
            {"color": [45, 89, 255], "id": 40, "name": "baseball glove"},
            {"color": [134, 134, 103], "id": 41, "name": "skateboard"},
            {"color": [145, 148, 174], "id": 42, "name": "surfboard"},
            {"color": [255, 208, 186], "id": 43, "name": "tennis racket"},
            {"color": [197, 226, 255], "id": 44, "name": "bottle"},
            {"color": [171, 134, 1], "id": 46, "name": "wine glass"},  # 40
            {"color": [109, 63, 54], "id": 47, "name": "cup"},
            {"color": [207, 138, 255], "id": 48, "name": "fork"},
            {"color": [151, 0, 95], "id": 49, "name": "knife"},
            {"color": [9, 80, 61], "id": 50, "name": "spoon"},
            {"color": [84, 105, 51], "id": 51, "name": "bowl"},
            {"color": [74, 65, 105], "id": 52, "name": "banana"},
            {"color": [166, 196, 102], "id": 53, "name": "apple"},
            {"color": [208, 195, 210], "id": 54, "name": "sandwich"},
            {"color": [255, 109, 65], "id": 55, "name": "orange"},
            {"color": [0, 143, 149], "id": 56, "name": "broccoli"},  # 50
            {"color": [179, 0, 194], "id": 57, "name": "carrot"},
            {"color": [209, 99, 106], "id": 58, "name": "hot dog"},
            {"color": [5, 121, 0], "id": 59, "name": "pizza"},
            {"color": [227, 255, 205], "id": 60, "name": "donut"},
            {"color": [147, 186, 208], "id": 61, "name": "cake"},
            {"color": [153, 69, 1], "id": 62, "name": "chair"},
            {"color": [3, 95, 161], "id": 63, "name": "couch"},
            {"color": [163, 255, 0], "id": 64, "name": "potted plant"},
            {"color": [119, 0, 170], "id": 65, "name": "bed"},
            {"color": [0, 182, 199], "id": 67, "name": "dining table"},  # 60
            {"color": [0, 165, 120], "id": 70, "name": "toilet"},
            {"color": [183, 130, 88], "id": 72, "name": "tv"},
            {"color": [95, 32, 0], "id": 73, "name": "laptop"},
            {"color": [130, 114, 135], "id": 74, "name": "mouse"},
            {"color": [110, 129, 133], "id": 75, "name": "remote"},
            {"color": [166, 74, 118], "id": 76, "name": "keyboard"},
            {"color": [219, 142, 185], "id": 77, "name": "cell phone"},
            {"color": [79, 210, 114], "id": 78, "name": "microwave"},
            {"color": [178, 90, 62], "id": 79, "name": "oven"},
            {"color": [65, 70, 15], "id": 80, "name": "toaster"},  # 70
            {"color": [127, 167, 115], "id": 81, "name": "sink"},
            {"color": [59, 105, 106], "id": 82, "name": "refrigerator"},
            {"color": [142, 108, 45], "id": 84, "name": "book"},
            {"color": [196, 172, 0], "id": 85, "name": "clock"},
            {"color": [95, 54, 80], "id": 86, "name": "vase"},
            {"color": [128, 76, 255], "id": 87, "name": "scissors"},
            {"color": [201, 57, 1], "id": 88, "name": "teddy bear"},
            {"color": [246, 0, 122], "id": 89, "name": "hair drier"},
            {"color": [191, 162, 208], "id": 90, "name": "toothbrush"},
            {"color": [0, 0, 0], "id": -1, "name": "background"},
        ]
        self.coco91to80 = {}
        for i, m in enumerate(self.meta):
            self.coco91to80[m["id"]] = i

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, key):
        if isinstance(key, slice):
            return [self.meta[ii] for ii in range(*key.indices(len(self.meta)))]
        elif isinstance(key, int):
            if key < 0:  # Handle negative indices
                key += len(self)
            if key < 0 or key >= len(self.meta):
                raise IndexError
            return self.meta[key]  # Get the data from elsewhere
        elif isinstance(key, str):
            for m in self.meta:
                if m["name"] == key:
                    return m
            raise KeyError
        else:
            raise TypeError

    def coco80to91(self, cls_id):
        return self.meta[cls_id]["id"]

    def __call__(self, cls_id):
        return self.meta[cls_id]


class VOCMeta:
    def __init__(self):
        self.meta = [
            {"color": [106, 0, 228], "id": 0, "name": "aeroplane"},
            {"color": [119, 11, 32], "id": 1, "name": "bicycle"},
            {"color": [165, 42, 42], "id": 2, "name": "bird"},
            {"color": [0, 0, 192], "id": 3, "name": "boat"},
            {"color": [197, 226, 255], "id": 4, "name": "bottle"},
            {"color": [0, 60, 100], "id": 5, "name": "bus"},
            {"color": [0, 0, 142], "id": 6, "name": "car"},
            {"color": [255, 77, 255], "id": 7, "name": "cat"},
            {"color": [153, 69, 1], "id": 8, "name": "chair"},
            {"color": [120, 166, 157], "id": 9, "name": "cow"},
            {"color": [0, 182, 199], "id": 10, "name": "diningtable"},
            {"color": [0, 226, 252], "id": 11, "name": "dog"},
            {"color": [182, 182, 255], "id": 12, "name": "horse"},
            {"color": [0, 0, 230], "id": 13, "name": "motorbike"},
            {"color": [220, 20, 60], "id": 14, "name": "person"},
            {"color": [163, 255, 0], "id": 15, "name": "pottedplant"},
            {"color": [0, 82, 0], "id": 16, "name": "sheep"},
            {"color": [100, 80, 100], "id": 17, "name": "sofa"},
            {"color": [0, 80, 100], "id": 18, "name": "train"},
            {"color": [183, 130, 88], "id": 19, "name": "tv"},
        ]

    def __len__(self):
        return len(self.meta)

    def __call__(self, cls_id):
        return self.meta[cls_id]

    def __getitem__(self, key):
        if isinstance(key, slice):
            return [self.meta[ii] for ii in range(*key.indices(len(self.meta)))]
        elif isinstance(key, int):
            if key < 0:  # Handle negative indices
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError
            return self.meta[key]  # Get the data from elsewhere
        elif isinstance(key, str):
            for m in self.meta:
                if m["name"] == key:
                    return m
            raise KeyError
        else:
            raise TypeError


cocometa = COCOMeta()
vocmeta = VOCMeta()
