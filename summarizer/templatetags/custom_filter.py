from atexit import register
from django import template

register = template.Library()


@register.filter(name="model_measure_way")
def model_measure_way(measure_way,label):
    if measure_way == label:
        return "checked"
    return ''