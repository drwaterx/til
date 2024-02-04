import altair as alt
import pandas
from pathlib import Path
from jinja2 import Template


def build_chart():
    source = pandas.DataFrame({'x': [0, 1, 2, 3, 4],
                               'y': [0.1, 0.4, 0.3, 0.2, 0.8]})
    chart = alt.Chart(source).mark_point().encode(
        x='x',
        y='y'
    )
    return chart


if __name__ == '__main__':
    chart1 = build_chart()
    p = Path.cwd() / 'altair_test3.md'
    template = Template(p.read_text())
    o = Path.cwd() / 'post_altair.html'
    o.write_text(template.render({'alt': alt,
                                  'dchart1': chart1.to_json(indent=None)})
                 )
