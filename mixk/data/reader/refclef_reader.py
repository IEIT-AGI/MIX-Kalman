from .refcoco2_reader import RefCOCOReader


class RefClefReader(RefCOCOReader):

    def __init__(self, cfg):
        super().__init__(cfg)

    def load_image_from_ref(self, ref):
        raise NotImplementedError
