# Define Annotation Class        
class Annotation:
    def __init__(self, tokens, label):
        self.tokens = tokens
        self.label = label

    def print_annotation(self):
        print(self.tokens)
        print(self.label)


# Define BertContainer Class
class BertContainer:
    def __init__(self, key, annotator, sen_id, sen, encoding):
        self.key = key
        self.annotator = annotator
        self.sen_id = sen_id
        self.sen = sen
        self.encoding = encoding
        self.annot = []

    def add_anno(self, anno):
        self.annot.append(anno)

    def print_container(self):
        info = []
        print(self.key)
        print(self.annotator)
        print(self.sen_id)
        print(self.sen)
        for anno in self.annot:
            anno.print_annotation()
        print(self.encoding)

    def write_to_pkl(self):
        self.dictionary = {self.key: [('annotator', self.annotator), ('sen_id', self.sen_id), ('sen', self.sen),
                                      ('annotations', self.annot), ('encoding', self.encoding)]}
        return (self.dictionary)
