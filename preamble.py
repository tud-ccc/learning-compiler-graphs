import base64
import IPython
import json


def view_dot(dot):
    IPython.display.display(IPython.display.Image(dot.create_png()))


def view_dots(dots):
    s = '<table><tr>'
    for dot in dots:
        s += '<th><img src="' + 'data:image/png;base64,' + base64.b64encode(dot.create_png()).decode(
            'ascii') + '"/></th>'
    s += '</tr></table>'

    t = IPython.display.HTML(s)
    IPython.display.display(t)