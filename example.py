# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "altair==5.5.0",
#     "anywidget==0.9.18",
#     "dicekit==0.1.1",
#     "marimo",
#     "mohtml==0.1.11",
#     "numpy==2.3.2",
#     "pandas==2.3.2",
#     "polars==1.32.3",
#     "pyobsplot==0.5.4",
#     "traitlets==5.14.3",
# ]
# ///

import marimo

__generated_with = "0.15.0"
app = marimo.App(width="full")


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    # Matrix as a User Interface

    This notebook is all about approaching a matrix as a user-interface element and how this allows you to train your intuition in a new way. Instead of "dedicated practice", you might call it "dedicated play". 

    ## How does it work 

    The entire notebook revolves around the idea of this matrix widget. You can see an example below.
    """
    )
    return


@app.cell
def _(Matrix, mo):
    demo_mat = mo.ui.anywidget(Matrix([[1, 2], [2, 1]]))
    demo_mat
    return (demo_mat,)


@app.cell
def _(mo):
    mo.md(r"""You can slide around and change each number. And as a result the underlying Python object also looks different.""")
    return


@app.cell
def _(demo_mat, np):
    np.array(demo_mat.value["matrix"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""But the real joy is in how you can mix and match them!""")
    return


@app.cell(hide_code=True)
def _(br, matmul, mo, sizes):
    mo.vstack([sizes, br(), matmul])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## An example of "play"

    Let's do something fun with these matrices. We're going to build a matrix that performs [PCA](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html). We are going to take a dataset that has three columns and we are going to control the matrix that turns them into two columns. 

    The matrix multiplication below shows how this might work for a single datapoint.
    """
    )
    return


@app.cell(hide_code=True)
def _(br, mat_c, mat_pca, mo, out, p):
    mo.vstack([
        br(), 
        br(), 
        mo.hstack(
            [   
                mat_c,
                p("", style="width: 40px"),
                mo.md(r"$$ {\Large \times }$$"),
                mat_pca,
                p("", style="width: 30px"),
                mo.md(r"$$ {\Large = }$$"),
                p("", style="width: 20px"),
                out,
            ],
            justify="start",
            wrap=True,
        ),
        br(), 
        br()
    ])
    return


@app.cell(hide_code=True)
def _(alt, color, mo, pca_mat, pd, rgb_mat):
    X_tfm = rgb_mat @ pca_mat.matrix
    df_pca = pd.DataFrame({"x": X_tfm[:, 0], "y": X_tfm[:, 1], "c": color})

    pca_chart = (
        alt.Chart(df_pca)
            .mark_point()
            .encode(x="x", y="y", color=alt.Color('c:N', scale = None))
            .properties(width=400, height=400)
    )

    mo.hstack([pca_mat, pca_chart])
    return


@app.cell(hide_code=True)
def _():
    # mat_mul = mo.ui.anywidget(
    #     Matrix(
    #         [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
    #         max_value=2.0,
    #         step=0.1,
    #     )
    # )
    return


@app.cell(hide_code=True)
def _(Matrix, mo, np):
    import pandas as pd
    import altair as alt

    pca_mat = mo.ui.anywidget(
        Matrix(np.random.normal(0, 1, size=(3, 2)), step=0.1, row_names=["R", "G", "B"])
    )
    rgb_mat = np.random.randint(0, 255, size=(1000, 3))
    color = ["#{0:02x}{1:02x}{2:02x}".format(r, g, b) for r,g,b in rgb_mat]

    rgb_df = pd.DataFrame({
        "r": rgb_mat[:, 0], "g": rgb_mat[:, 1], "b": rgb_mat[:, 2], 'color': color
    })
    return alt, color, pca_mat, pd, rgb_mat


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Final demo, changing a coordinate""")
    return


@app.cell
def _(np):
    angle = np.pi / 12
    return (angle,)


@app.cell
def _(Matrix, angle, mo, np):
    mat_mul = mo.ui.anywidget(
        Matrix(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
            max_value=2.0,
            step=0.1,
        )
    )
    return (mat_mul,)


@app.cell
def _(Matrix, mo):
    mat_point = mo.ui.anywidget(Matrix([[1, 2]], max_value=2.0, step=0.1))
    return (mat_point,)


@app.cell
def _(mo):
    step_slider = mo.ui.slider(1, 200, 1, label="Number of matrix multiplies")
    return (step_slider,)


@app.cell(hide_code=True)
def _(arrow_plot, mat_mul, mat_point, mo, p, step_slider):
    latex = mo.hstack(
        [
            mat_point,
            p("", style="width: 20px"),
            mo.md(r"$$ {\Large \times }$$"),
            p("", style="width: 20px"),
            mat_mul,
            p("", style="width: 20px"),
        ],
        justify="start",
        align="center",
        wrap=True,
    )

    mo.hstack([latex, mo.vstack([step_slider, arrow_plot])])
    return


@app.cell(hide_code=True)
def _(mat_mul, mat_point, np, pl, step_slider):
    orig_point = np.array(mat_point.matrix)
    data = [orig_point]
    for i in range(step_slider.value):
        orig_point = orig_point @ np.array(mat_mul.matrix)
        data.append(orig_point)
    df_forward = pl.DataFrame(np.array(data)[:, 0, :], schema=["x", "y"])
    return (df_forward,)


@app.cell(hide_code=True)
def _(df_forward, pl):
    df_arrows = (
        df_forward.with_columns(y2=pl.col("y").shift(-1), x2=pl.col("x").shift(-1))
        .rename(dict(x="x1", y="y1"))
        .drop_nulls()
    )
    return (df_arrows,)


@app.cell(hide_code=True)
def _(Plot, df_arrows, df_forward, pl):
    metros = pl.DataFrame(
        [{"x1": 0, "x2": 1, "y1": 1, "y2": 2}, {"x1": 1, "x2": 1.5, "y1": 2, "y2": 3}]
    )

    arrow_plot = Plot.plot(
        {
            "height": 400,
            "width": 400,
            "grid": True,
            "inset": 10,
            "x": {"label": "x →"},
            "y": {"label": "↑ y", "ticks": 4},
            "marks": [
                Plot.arrow(
                    df_arrows,
                    {
                        "x1": "x1",
                        "y1": "y1",
                        "x2": "x2",
                        "y2": "y2",
                        "bend": False,
                        "stroke": "gray",
                    },
                ),
                Plot.dot(df_forward, {"x": "x", "y": "y", "fill": "black"}),
            ],
        }
    )
    return (arrow_plot,)


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Appendix""")
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import polars as pl
    from pyobsplot import Plot
    return Plot, mo, np, pl


@app.cell
def _(mo):
    from wigglystuff import Matrix

    mat_pca = mo.ui.anywidget(
        Matrix([[1, 2], [1, 3], [3, 1]], row_names=["R", "G", "B"], max_value=100)
    )
    mat_c = mo.ui.anywidget(Matrix([[1, 2, 3]], col_names=["R", "G", "B"]))
    return Matrix, mat_c, mat_pca


@app.cell
def _(Matrix, mat_c, mat_pca, mo, np):
    out = mo.ui.anywidget(
        Matrix(np.array(mat_c.matrix) @ np.array(mat_pca.matrix), static=True, col_names=["x", "y"])
    )
    return (out,)


@app.cell
def _():
    from mohtml import p, br
    return br, p


@app.cell
def _(mo):
    a_rows = mo.ui.slider(1, 5, 1, label="Rows for matrix A")
    a_cols = mo.ui.slider(1, 5, 1, label="Columns for matrix A")
    b_cols = mo.ui.slider(1, 5, 1, label="Columns for matrix B")
    return a_cols, a_rows, b_cols


@app.cell
def _(Matrix, mo, np, sizes):
    ar, ac, bc = sizes.value.values()

    mat_a = mo.ui.anywidget(Matrix(np.ones(shape=(ar, ac))))
    mat_b = mo.ui.anywidget(Matrix(np.ones(shape=(ac, bc))))
    return mat_a, mat_b


@app.cell
def _(Matrix, mat_a, mat_b, mo, np):
    np_out = np.array(mat_a.matrix) @ np.array(mat_b.matrix)
    mat_out = mo.ui.anywidget(Matrix(np_out, static=True))
    return (mat_out,)


@app.cell
def _(a_cols, a_rows, b_cols, mo):
    sizes = mo.md("""
    ### Playing with sizes

    We can add more "native" web inputs to help create these matrices. Below we're using sliders to toy around with the shapes of matrices. 

    {a_rows} {a_cols} {b_cols}
    """).batch(a_rows=a_rows, a_cols=a_cols, b_cols=b_cols)
    return (sizes,)


@app.cell
def _(mat_a, mat_b, mat_out, mo, p):
    matmul = mo.hstack(
        [
            mat_a,
            p("", style="width: 20px"),
            mo.md(r"$$ {\Large \times }$$"),
            p("", style="width: 20px"),
            mat_b,
            p("", style="width: 30px"),
            mo.md(r"$$ {\Large = }$$"),
            p("", style="width: 20px"),
            mat_out,
        ],
        justify="start",
        align="center",
        wrap=True,
    )
    return (matmul,)


@app.cell
def _():
    from anywidget import AnyWidget
    import traitlets
    return


if __name__ == "__main__":
    app.run()
