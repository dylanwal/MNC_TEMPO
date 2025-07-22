
import numpy as np
from scipy.integrate import solve_ivp

import expt_data.data as expt_data
from constants import get_const
from ode_function import odefunc


def solve_ODE(data_: expt_data.Data, k: np.ndarray, t: np.ndarray = None) -> np.ndarray:
    const = get_const(data_.H, data_.O2, data_.Fe)
    y0 = np.array([
            data_.TEMPO_p[0] if data_.TEMPO_p[0] != 0 else 0.0000001,  # TEMPO_p
            data_.TEMPO[0] if data_.TEMPO[0] != 0 else 0.0000001,  # TEMPO
            data_.TEMPOH2_p[0] if data_.TEMPOH2_p[0] != 0 else 0.0000001,  # TEMPOH2
        ])
    sol = solve_ivp(odefunc, data_.t_span, y0, args=(k, const), method='Radau', t_eval=t if t is not None else data_.time)
    return sol.y.T


def get_data(id_: int) -> expt_data.Data:
    for d in expt_data.ALL_DATA_CUT:
        if d.id_ == id_:
            return d
    raise ValueError("no data found")


def main():
    from plotly.subplots import make_subplots

    figs = [[1, 5, 12], [14, 3,10],[8,13,15]]
    titles = []
    for row in figs:
        for col in row:
            titles.append(str(get_data(col)))

    fig = make_subplots(rows=3, cols=3, subplot_titles=titles)
    for col, i in enumerate(figs):
        col += 1
        for row, ii in enumerate(i):
            row += 1
            data_ = get_data(ii)
            # k = np.array([971.346830197743,0.5802317286184792,8.158086014865025e-05,10000000000.0,71.71640322492323, 63.23939458995973])
            k = np.array([72.42636163236209,0.6836457945967959,4.802251991995693e-09,62192.66772280553,0.6498547141709465,1e-05])
            # k = np.array([7.74104752207566, 0.862891173493384, 8.85116656134E-09, 58183.0599974819, 0.000962922800758801, 0.00022852474646638]) # excel
            t = np.linspace(data_.t_span[0], data_.t_span[1], 200)
            result = solve_ODE(data_, k,t)

            # plot fits
            fig.add_scatter(row=col, col=row, x=t, y=result[:, 0], name="TEMPO_p", mode="lines", line=dict(color="blue"))
            fig.add_scatter(row=col, col=row, x=t, y=result[:, 1], name="TEMPO", mode="lines", line=dict(color="goldenrod"))
            fig.add_scatter(row=col, col=row, x=t, y=result[:, 2], name="TEMPOH2", mode="lines", line=dict(color="red"))

            # plot data
            fig.add_scatter(row=col, col=row,x=data_.time, y=data_.TEMPOH2_p, name="TEMPOH2", mode="markers", marker=dict(color="red"))
            fig.add_scatter(row=col, col=row,x=data_.time, y=data_.TEMPO, name="TEMPO", mode="markers", marker=dict(color="goldenrod"))
            fig.add_scatter(row=col, col=row,x=data_.time, y=data_.TEMPO_p, name="TEMPO_p", mode="markers", marker=dict(color="blue"))

            # update layout
            fig.layout.title.text = f"{data_.id_}, pH: {data_.pH}, Fe: {data_.Fe}, O2: {data_.O2}"
            fig.update_xaxes(title_text="time", row=col, col=row)
            fig.update_yaxes(title_text="concentration", row=col, col=row)

            # save fits
            np.savetxt(
                fr"..\fit_data\{data_}.csv",
                np.column_stack((t, result)),
                delimiter=",",
                header="time(s), TEMPO+, TEMPO, TEMPOH2+"
            )

    fig.update_layout(showlegend=False)
    fig.show()


if __name__ == "__main__":
    main()
