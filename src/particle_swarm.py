from typing import Callable, Iterable
import logging
import multiprocessing as mp
import copy

import numpy as np

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)
handler.setLevel(logging.DEBUG)


class Particle:
    _ID_ = 0

    def __init__(self, x: float, v: float):
        self.i = -1
        self._x = []
        self._v = []
        self._obj = []
        self.set_xv(x, v)
        self.alive = True
        self.id_ = Particle._ID_
        Particle._ID_ += 1

    def __str__(self):
        return f"id: {self.id_} | {self.x} (v: {self.v}) obj: {self._obj}"

    @property
    def x(self):
        return self._x[self.i]

    @property
    def v(self):
        return self._v[self.i]

    @property
    def obj(self):
        try:
            return self._obj[self.i]
        except IndexError:
            raise ValueError("Particle has not computed objective yet.")

    @property
    def x_best(self):
        return self._x[np.argmin(self._obj)]

    @property
    def obj_best(self):
        return self._obj[np.argmin(self._obj)]

    def set_xv(self, x: float | np.ndarray, v: float | np.ndarray):
        self._x.append(x)
        self._v.append(v)
        self.i += 1

    def set_obj(self, obj: float):
        self._obj.append(obj)


class Swarm(list):
    def iter_alive(self) -> Iterable[Particle]:
        for particle in self:
            if particle.alive:
                yield particle

    def number_alive(self) -> int:
        return sum(1 for particle in self if particle.alive)


def _rescale(x: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    span = bounds[1] - bounds[0]
    return (x * span) + bounds[0]


def _calculate_starting_position_and_velocity(
        number_of_particles: int,
        bounds: np.ndarray,
        rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    from scipy.stats import qmc

    dim = bounds.shape[0] // 2
    assert bounds.shape == (2 * dim,), "Bounds should be a flat array of shape (2 * dim,)"

    sampler = qmc.Sobol(d=dim, seed=rng)
    n_samples = 2 ** int(np.ceil(np.log2(number_of_particles)))
    x_unit = sampler.random_base2(m=int(np.log2(n_samples)))
    x_unit = x_unit[:number_of_particles]

    # Rescale to bounds using vectorized transformation
    bounds = bounds.reshape(dim, 2)
    lower, upper = bounds[:, 0], bounds[:, 1]
    x = lower + (upper - lower) * x_unit  # broadcasted rescaling

    v = rng.normal(loc=0, scale= 0.5, size=(number_of_particles, dim))
    return x, v


def _compute_objective(swarm: Swarm, objective: Callable, args: tuple = None, multiprocess: bool | int = False):
    if multiprocess:
        if isinstance(multiprocess, int):
            cpu = min(mp.cpu_count(), multiprocess)
        else:
            cpu = None
        args_ = [[particle.x] + list(args) for particle in swarm.iter_alive()]
        with mp.Pool(processes=cpu) as pool:
            result = pool.starmap(objective, args_)

        for particle, r in zip(swarm.iter_alive(), result):
            particle.set_obj(r)
        return

    for particle in swarm.iter_alive():
        particle.set_obj(objective(particle.x, *args))


def _get_best_particle(swarm: Swarm) -> Particle:
    best_particle = swarm[0]
    for particle in swarm.iter_alive():
        if particle.obj < best_particle.obj:
            best_particle = particle
    return best_particle


def _remove_similar_particles(swarm: Swarm, min_distance: float = 0.001):
    if swarm.number_alive() <= 1:
        return

    particles = list(swarm.iter_alive())
    position = np.array([np.hstack((p.x, p.v)) for p in particles])
    normalized_position = (position - np.mean(position, axis=0)) / np.std(position, axis=0)

    alive_mask = np.ones(len(particles), dtype=bool)

    for i in range(len(particles)):
        if not alive_mask[i]:
            continue
        dists = np.linalg.norm(normalized_position[i] - normalized_position, axis=1)
        # Ignore self by setting a high value at self index
        dists[i] = np.inf
        j = np.argmin(dists)
        if dists[j] < min_distance:
            alive_mask[j] = False

    for i, keep in enumerate(alive_mask):
        particles[i].alive = keep


def _check_bounds(x: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    """Clamp x to stay within the provided bounds (flat array format: [low0, high0, low1, high1, ...])."""
    dim = len(x)
    bounds = bounds.reshape(dim, 2)
    return np.clip(x, bounds[:, 0], bounds[:, 1])


def particle_swarm_optimization(
        objective: Callable,
        number_of_particles: int,
        bounds: np.ndarray,
        args: tuple = None,
        max_iter: int = 100,
        w: float = 0.5,
        c1: float = 1.5,
        c2: float = 1.5,
        rng: np.random.Generator = np.random.default_rng(seed=0),
        stagnation_iters: int = 5,
        stagnation_tol: float = 1e-6,
        pruning_distance: float = 1,
        output: str | None = None,
        logging_level: int = logging.INFO,
        multiprocess: bool | int = False
) -> np.ndarray | tuple[np.ndarray, Particle] | tuple[np.ndarray, list[Particle, ...]]:
    """
    Particle swarm optimization (minimization)
    - with pruning

    Parameters
    ----------
    objective:
        problem you are trying to solve
    number_of_particles:
        number of particles to spawn
    bounds:
        bounds of exploration
    args:
        args to pass to objective function
    max_iter:
        max number of iterations
    w:
        the inertia weight constant; determines how much should the particle keep on with its previous velocity
        between 0 and 1
    c1:
        typically between 0.1 and 3
        cognitive coefficients; exploration
        weight should be given to refining the search result of the particle itself
    c2:
        typically between 0.1 and 3typically between 0.1 and 3
        social coefficients;exploitation
        weight should be given to recognizing the search result of the swarm
    rng:
        random number generator
    stagnation_iters:
        number of stagnation iterations before the stopping criterion is met
    stagnation_tol:
        the tolerance for a stagnation iteration to be considered
    pruning_distance:
        distance between the particles before it is pruned
    output:
        None: x values
        "best particle": x values and Particle object
        "swarm": list of all Particle objects
    logging_level:
        output level for logging
    multiprocess:
        allow multiple cores to be used
        True: will use all cores
        int: will use specified number of cores

    Returns
    -------
    depending on output parameter


    """
    args = args or ()
    logger.setLevel(logging_level)

    x, v = _calculate_starting_position_and_velocity(number_of_particles, bounds, rng)
    swarm = Swarm(Particle(x_, v_) for x_, v_ in zip(x, v))

    logger.info(f"step | objective | id | best values  | # of alive particles")
    logger.info("-" * 100)

    _compute_objective(swarm, objective, args, multiprocess)
    best_particle = _get_best_particle(swarm)

    logger.debug(f"-1 | {best_particle.obj:0.6e} | {best_particle.id_} | {best_particle.x} | {swarm.number_alive()}")

    stagnation_count = 0
    prev_best_particle = copy.deepcopy(best_particle)
    for i in range(max_iter):
        # update position and velocity
        r1, r2 = rng.random(2)
        best_particle_x = prev_best_particle.x_best
        for particle in swarm.iter_alive():
            v = w * particle.v + c1 * r1 * (particle.x_best - particle.x) + c2 * r2 * (best_particle_x - particle.x)
            x = particle.x + v
            x = _check_bounds(x, bounds)
            particle.set_xv(x, v)

        _compute_objective(swarm, objective, args, multiprocess)

        best_particle = _get_best_particle(swarm)
        # stopping criteria (Stagnation)
        rel_improvement = abs(best_particle.obj - prev_best_particle.obj_best)/abs(prev_best_particle.obj_best + 1e-16) < stagnation_tol
        if rel_improvement and i > stagnation_iters + 1:
            stagnation_count += 1
            if stagnation_count >= stagnation_iters:
                logger.info(f"Stagnation stopping criteria met! "
                      f"\n\tNo improvement over {stagnation_iters} iterations at iteration {i}")
                break
        else:
            stagnation_count = 0

        # update best
        if best_particle.obj < prev_best_particle.obj_best:
            prev_best_particle = copy.deepcopy(best_particle)

        if swarm.number_alive() == 1:
            logger.info(f"Swarm has converged to a single particle.")
            break

        # optimization
        _remove_similar_particles(swarm, pruning_distance)

        logger.debug(f"{i} | {prev_best_particle.obj:0.6e} | {prev_best_particle.id_} | {prev_best_particle.x} | {swarm.number_alive()}")

    if output == "swarm":
        return best_particle.x, swarm
    if output == "best particle":
        return best_particle.x, best_particle
    return best_particle.x


def run_local():
    def _obj(x_: np.array) -> float | np.ndarray:
        x = x_[0]
        y = x_[1]
        return 0.2 * ((x - 3.14) ** 2 + (y - 2.72) ** 2 + np.sin(3 * x + 1.41) + np.sin(4 * y - 1.73))

    true_min = (3.1852, 3.1298)
    bounds = np.array([0, 5, 0, 5])
    PSO_min, swarm = particle_swarm_optimization(
        _obj,
        25,
        bounds,
        c1=0.5,
        c2=0.5,
        pruning_distance=1,
        output="swarm",
        logging_level=logging.DEBUG
    )

    # compute global_brute force
    n = 50
    x = np.linspace(0, 5, n)
    y = np.linspace(0, 5, n)
    xx, yy = np.array(np.meshgrid(x, y))
    z = np.array([_obj(row) for row in np.column_stack((xx.ravel(), yy.ravel()))])
    x_min = xx.ravel()[z.argmin()]
    y_min = yy.ravel()[z.argmin()]

    # outputs
    print(f"True min: ({true_min[0]}, {true_min[1]})")
    print(f"Grid min: ({x_min:0.4f}, {y_min:0.4f})")
    print(f"PSO min: ({PSO_min[0]:0.4f}, {PSO_min[1]:0.4f})")

    import plotly.graph_objs as go

    fig = go.Figure()
    fig.add_surface(x=x, y=y, z=z.reshape((len(x), len(y))), showscale=False)
    fig.add_scatter3d(x=[true_min[0]], y=[true_min[1]], z=[_obj(true_min)], mode='markers', name='true min')
    fig.add_scatter3d(x=[PSO_min[0]], y=[PSO_min[1]], z=[_obj(PSO_min)], mode='markers', name='PSO min')
    for ii, particle in enumerate(swarm):
        x = np.array(particle._x)
        fig.add_scatter3d(x=x[:, 0], y=x[:, 1], z=particle._obj + np.abs(np.min(particle._obj)) * 0.05,
                          mode='markers+lines',
                          name='swarm', legendgroup='swarm', showlegend=True if ii == 0 else False,
                          marker={"size": 3})
    fig.show()

    # from plotly_gif import GIF
    # gif = GIF()
    #
    # for i in range(len(swarm[0]._x)):
    #     fig = go.Figure()
    #     fig.add_surface(x=x, y=y, z=z.reshape((len(x), len(y))), colorbar=False)
    #     fig.add_scatter3d(x=[true_min[0]], y=[true_min[1]], z=[obj(true_min)], mode='markers', name='true min')
    #     fig.add_scatter3d(x=[PSO_min[0]], y=[PSO_min[1]], z=[obj(PSO_min)], mode='markers', name='PSO min')
    #
    #     for ii, particle in enumerate(swarm):
    #         fig.add_scatter3d(x=[particle._x[i][0]], y=[particle._x[i][1]], z=[particle._obj[i]], mode='markers', name='best particle', legendgroup='swarm', showlegend=True if ii == 0 else False)
    #         ax = particle._x[i][0] + particle._v[i][0]
    #         ay = particle._x[i][1] + particle._v[i][1]
    #         fig.add_scatter3d(x=[particle._x[i][0], ax], y=[particle._x[i][1], ay], z=[particle._obj[i], obj(np.array([ax, ay]))], mode='lines', name='best particle', legendgroup='swarm', showlegend=False)
    #     gif.create_image(fig)
    #
    # gif.create_gif()


if __name__ == '__main__':
    run_local()
