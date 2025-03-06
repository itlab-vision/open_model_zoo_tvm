from ..representation import ClassificationAnnotation
from pathlib import Path
from .format_converter import GraphFileBasedAnnotationConverter, ConverterReturn

try:
    from spektral.utils.io import load_binary  # pylint: disable=C0415
except ImportError as import_error:
    raise ValueError(
        "Spektral isn't installed. Please, install it before using. \n{}".format(
            import_error.msg))
try:
    import tensorflow as tf  # pylint: disable=C0415
except ImportError as import_error:
    raise ValueError(
        "Tensorflow isn't installed. Please, install it before using. \n{}".format(
            import_error.msg))


class SpektralConverter(GraphFileBasedAnnotationConverter):
    __provider__ = "spektral_converter"

    def convert(self, check_content=False, **kwargs):
        graph = load_binary(Path(self.graph_path).__str__())

        labels = tf.math.argmax(graph.y, 1)

        annotation = [
            ClassificationAnnotation(identifier='', label=labels)
        ]

        return ConverterReturn(annotation, {'labels': labels}, None)