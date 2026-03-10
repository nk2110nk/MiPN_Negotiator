import time
from typing import List, Union, Tuple, Optional
from negmas.mechanisms import MechanismRoundResult
from negmas.outcomes import (
    ResponseType,
    outcome_is_complete,
)
from negmas.sao import SAOMechanism, SAONegotiator, SAOResponse
import itertools
import pandas as pd
import math


class MySAOMechanism(SAOMechanism):
    def __init__(self, **kwargs):
        super().__init__(keep_issue_names=True, end_on_no_response=True, **kwargs)

    def reset(self):
        for neg in self._negotiators:
            neg.reset()
        del self.ami
        del self._current_proposer
        del self.agents_of_role
        del self._negotiators
        del self._history
        del self._Mechanism__outcome_index
        del self._Mechanism__outcomes
        del self._state_factory

    def round(self) -> MechanismRoundResult:
        self._new_offers = []
        negotiators: List[SAONegotiator] = self.negotiators
        n_negotiators = len(negotiators)

        def _safe_counter(negotiator, *args, **kwargs):
            try:
                if negotiator == self._current_proposer and self._offering_is_accepting:
                    self._n_accepting = 0
                    kwargs["offer"] = None
                    response = negotiator.counter(*args, **kwargs)
                else:
                    response = negotiator.counter(*args, **kwargs)
            except Exception as ex:
                if self.ignore_negotiator_exceptions:
                    return SAOResponse(ResponseType.END_NEGOTIATION, None)
                else:
                    raise ex
            if (
                self.check_offers
                and response.outcome is not None
                and (not outcome_is_complete(response.outcome, self.issues))
            ):
                return SAOResponse(response.response, None)
            return response

        proposers, proposer_indices = [], []
        for i, neg in enumerate(negotiators):
            if not neg.capabilities.get("propose", False):
                continue
            proposers.append(neg)
            proposer_indices.append(i)
        n_proposers = len(proposers)

        # this is not the first round. A round will get n_negotiators steps

        self._last_checked_negotiator = (self._last_checked_negotiator + 1) % n_negotiators
        neg = self.negotiators[self._last_checked_negotiator]
        strt = time.perf_counter()
        resp = _safe_counter(neg, state=self.state, offer=self._current_offer)
        if time.perf_counter() - strt > self.ami.step_time_limit:
            return MechanismRoundResult(broken=False, timedout=True, agreement=None)
        if resp.response == ResponseType.END_NEGOTIATION:
            return MechanismRoundResult(broken=True, timedout=False, agreement=None)
        if resp.response == ResponseType.ACCEPT_OFFER:
            self._n_accepting += 1
            if self._n_accepting == n_negotiators:
                return MechanismRoundResult(
                    broken=False, timedout=False, agreement=self._current_offer
                )
        if resp.response == ResponseType.REJECT_OFFER:
            proposal = resp.outcome
            if proposal is None:
                if (
                    neg.capabilities.get("propose", False)
                    and self.end_negotiation_on_refusal_to_propose
                ):
                    return MechanismRoundResult(
                        broken=True, timedout=False, agreement=None
                    )
            else:
                self._current_offer = proposal
                self._current_proposer = neg
                self._new_offers.append((neg.id, proposal))
                self._n_accepting = 1 if self._offering_is_accepting else 0
                if self._enable_callbacks:
                    for other in self.negotiators:
                        if other is neg:
                            continue
                        other.on_partner_proposal(
                            agent_id=neg.id, offer=proposal, state=self.state
                        )
        return MechanismRoundResult(broken=False, timedout=False, agreement=None)

    def plot(
        self,
        visible_negotiators: Union[Tuple[int, int], Tuple[str, str]] = (0, 1),
        plot_utils=True,
        plot_outcomes=False,
        utility_range: Optional[Tuple[float, float]] = None,
        path: Optional[str] = None
    ):
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        import matplotlib.ticker as tick

        if self.issues is not None and len(self.issues) > 1:
            plot_outcomes = False

        if len(self.negotiators) < 2:
            print("Cannot visualize negotiations with more less than 2 negotiators")
            return
        if len(visible_negotiators) > 2:
            print("Cannot visualize more than 2 agents")
            return
        if isinstance(visible_negotiators[0], str):
            tmp = []
            for _ in visible_negotiators:
                for n in self.negotiators:
                    if n.id == _:
                        tmp.append(n)
        else:
            visible_negotiators = [
                self.negotiators[visible_negotiators[0]],
                self.negotiators[visible_negotiators[1]],
            ]
        indx = dict(zip([_.id for _ in self.negotiators], range(len(self.negotiators))))
        history = []
        for state in self.history:
            for a, o in state.new_offers:
                history.append(
                    {
                        "current_proposer": a,
                        "current_offer": o,
                        "offer_index": self.outcomes.index(o),
                        "relative_time": state.relative_time,
                        "step": state.step,
                        "u0": visible_negotiators[0].utility_function(o),
                        "u1": visible_negotiators[1].utility_function(o),
                        "agreement": state.agreement,
                    }
                )
        history = pd.DataFrame(data=history)
        has_history = len(history) > 0
        has_front = 1
        n_negotiators = len(self.negotiators)
        n_agents = len(visible_negotiators)
        ufuns = self._get_ufuns()
        outcomes = self.outcomes
        utils = [tuple(f(o) for f in ufuns) for o in outcomes]
        agent_names = [a.name for a in visible_negotiators]
        if has_history:
            history["offer_index"] = [outcomes.index(_) for _ in history.current_offer]
        frontier, frontier_outcome = self.pareto_frontier(sort_by_welfare=True)
        frontier_indices = [
            i
            for i, _ in enumerate(frontier)
            if _[0] is not None
            and _[0] > float("-inf")
            and _[1] is not None
            and _[1] > float("-inf")
        ]
        frontier = [frontier[i] for i in frontier_indices]
        max_welfare = frontier[0]
        frontier = sorted(frontier, key=lambda x: x[0])
        frontier_outcome = [frontier_outcome[i] for i in frontier_indices]
        frontier_outcome_indices = [outcomes.index(_) for _ in frontier_outcome]
        if plot_utils:
            fig_util = plt.figure(figsize=(8.0, 4.8))
        if plot_outcomes:
            fig_outcome = plt.figure(figsize=(8.0, 4.8))
        gs_util = gridspec.GridSpec(n_agents, has_front * 3 + 2) if plot_utils else None
        gs_outcome = (
            gridspec.GridSpec(n_agents, has_front * 3 + 2) if plot_outcomes else None
        )
        axs_util, axs_outcome = [], []

        for a in range(n_agents):
            if a == 0:
                if plot_utils:
                    axs_util.append(fig_util.add_subplot(gs_util[a, -2:]))
                if plot_outcomes:
                    axs_outcome.append(
                        fig_outcome.add_subplot(gs_outcome[a, -2:])
                    )
            else:
                if plot_utils:
                    axs_util.append(
                        fig_util.add_subplot(gs_util[a, -2:], sharex=axs_util[0])
                    )
                if plot_outcomes:
                    axs_outcome.append(
                        fig_outcome.add_subplot(
                            gs_outcome[a, -2:], sharex=axs_outcome[0]
                        )
                    )
            if plot_utils:
                clrs = ("blue", "green")
                axs_util[-1].set_ylabel(agent_names[a] + "\'s Utility", color=clrs[a])
                # if a != 0:
                #     axs_util[-1].set_xlabel('Time')
                # if a == 0:
                #     axs_util[-1].set_title("Time-Util Graph")
            if plot_outcomes:
                axs_outcome[-1].set_ylabel(agent_names[a] + "\'s Utility")
        for a, (au, ao) in enumerate(
            zip(
                itertools.chain(axs_util, itertools.repeat(None)),
                itertools.chain(axs_outcome, itertools.repeat(None)),
            )
        ):
            if au is None and ao is None:
                break
            if has_history:
                h = history.loc[
                    history.current_proposer == visible_negotiators[a].id,
                    ["relative_time", "offer_index", "current_offer"],
                ]
                h["utility"] = h["current_offer"].apply(ufuns[a])
                h_opp = history.loc[
                    history.current_proposer != visible_negotiators[a].id,
                    ["relative_time", "offer_index", "current_offer"],
                ]
                h_opp["utility"] = h_opp["current_offer"].apply(ufuns[a])
                if plot_outcomes:
                    ao.plot(h.relative_time, h["offer_index"], marker=',')
                if plot_utils:
                    if a == 0:
                        clrs = ("blue", "green")
                    else:
                        clrs = ("green", "blue")
                    au.plot(h.relative_time, h.utility, marker=',', color=clrs[0])
                    au.plot(h_opp.relative_time, h_opp.utility, marker=',', color=clrs[1])
                    if utility_range is not None:
                        au.set_ylim(*utility_range)
                    else:
                        au.set_ylim(0.0, 1.05)

        if has_front:
            if plot_utils:
                axu = fig_util.add_subplot(gs_util[:, 0:-2])
                axu.scatter(
                    [_[0] for _ in utils],
                    [_[1] for _ in utils],
                    label="Outcomes",
                    color="pink",
                    marker=".",
                    s=10,
                    zorder=0
                )
            if plot_outcomes:
                axo = fig_outcome.add_subplot(gs_outcome[:, 0:-2])
            clrs = ("blue", "green")
            mkr = ('v', '^')
            if plot_utils:
                f1, f2 = [_[0] for _ in frontier], [_[1] for _ in frontier]
                # axu.scatter(f1, f2, label="frontier", color="red", marker="x")
                axu.plot(f1, f2, linewidth=1.0, label="Pareto", color="magenta", marker="o", markersize=6, zorder=1)
                # axu.legend(loc='lower left')
                axu.set_xlabel(agent_names[0] + "\'s Utility")
                axu.xaxis.label.set_color(clrs[0])
                axu.set_ylabel(agent_names[1] + "\'s Utility")
                axu.yaxis.label.set_color(clrs[1])
                # axu.set_title("Outcome Space")
                if self.agreement is not None:
                    pareto_distance = 1e9
                    cu = (ufuns[0](self.agreement), ufuns[1](self.agreement))
                    for pu in frontier:
                        dist = math.sqrt((pu[0] - cu[0]) ** 2 + (pu[1] - cu[1]) ** 2)
                        if dist < pareto_distance:
                            pareto_distance = dist
                    # axu.text(
                    #     0.03,
                    #     0.18,
                    #     f"Pareto-distance={pareto_distance:5.2}",
                    #     verticalalignment="top",
                    #     transform=axu.transAxes,
                    # )

            if plot_outcomes:
                axo.scatter(
                    frontier_outcome_indices,
                    frontier_outcome_indices,
                    color="magenta",
                    marker="o",
                    label="Pareto",
                    s=6,
                    zorder=1
                )
                axo.legend(loc='lower left')
                axo.set_xlabel(agent_names[0])
                axo.set_ylabel(agent_names[1])

            if plot_utils and has_history:
                axu.scatter(
                    [max_welfare[0]],
                    [max_welfare[1]],
                    color="black",
                    marker="D",
                    label=f"Nash",
                    zorder=2
                )

                for a in range(n_agents):
                    h = history.loc[
                        (history.current_proposer == self.negotiators[a].id)
                        | ~(history["agreement"].isnull()),
                        ["relative_time", "offer_index", "current_offer"],
                    ]
                    h["u0"] = h["current_offer"].apply(ufuns[0])
                    h["u1"] = h["current_offer"].apply(ufuns[1])

                    # make color map
                    import matplotlib.collections as mcoll
                    import matplotlib.path as mpath
                    from matplotlib.colors import LinearSegmentedColormap as lsc
                    import numpy as np

                    def colorline(x, y, cmap=plt.get_cmap('copper'), norm=plt.Normalize(0.0, 1.0)):
                        pth = mpath.Path(np.column_stack([h.u0, h.u1]))
                        verts = pth.interpolated(steps=3).vertices
                        x, y = verts[:, 0], verts[:, 1]
                        z = np.linspace(0, 1, len(x))
                        segments = make_segments(x, y)
                        lc = mcoll.LineCollection(
                            segments,
                            array=z,
                            cmap=cmap,
                            norm=norm,
                            linewidth=1,
                            zorder=3,
                        )
                        axu.add_collection(lc)
                        return lc

                    def make_segments(x, y):
                        if len(x) == 1:
                            return np.array([[(u0, u1) for u0, u1 in zip(x, y)]])
                        points = np.array([x, y]).T.reshape(-1, 1, 2)
                        segments = np.concatenate([points[:-1], points[1:]], axis=1)
                        return segments

                    cmap = lsc.from_list('colormap_name', ['light' + clrs[a], clrs[a]])
                    if len(h.u0) != 0:
                        lc = mcoll.LineCollection(
                            make_segments(h.u0, h.u1),
                            array=np.linspace(0, 1, len(h.u0)),
                            cmap=cmap,
                            norm=plt.Normalize(0.0, 1.0),
                            linewidth=1,
                            zorder=3,
                        )
                        axu.add_collection(lc)
                        # colorline(h.u0, h.u1, cmap=cmap)

                    axu.scatter(
                        h.u0,
                        h.u1,
                        c=np.linspace(0, 1, len(h.u0)),
                        marker=mkr[a],
                        cmap=cmap,
                        # label=f"{agent_names[a]}",
                        s=5**2,
                        zorder=3
                    )

                    # axu.plot(
                    #     h.u0,
                    #     h.u1,
                    #     linewidth=1,
                    #     marker=mkr[a],
                    #     color=clrs[a],
                    #     label=f"{agent_names[a]}",
                    #     markersize=5,
                    #     zorder=3
                    # )
            if plot_outcomes and has_history:
                steps = sorted(history.step.unique().tolist())
                aoffers = [[], []]
                for step in steps[::2]:
                    offrs = []
                    for a in range(n_agents):
                        a_offer = history.loc[
                            (history.current_proposer == agent_names[a])
                            & ((history.step == step) | (history.step == step + 1)),
                            "offer_index",
                        ]
                        if len(a_offer) > 0:
                            offrs.append(a_offer.values[-1])
                    if len(offrs) == 2:
                        aoffers[0].append(offrs[0])
                        aoffers[1].append(offrs[1])
                axo.scatter(aoffers[0], aoffers[1], color=clrs[0], label=f"offers")

            if self.state.agreement is not None:
                if plot_utils:
                    axu.scatter(
                        [ufuns[0](self.state.agreement)],
                        [ufuns[1](self.state.agreement)],
                        color="red",
                        marker="s",
                        s=50,
                        label="Agreement",
                        zorder=4
                    )
                    axu.legend(loc='lower left')
                if plot_outcomes:
                    axo.scatter(
                        [outcomes.index(self.state.agreement)],
                        [outcomes.index(self.state.agreement)],
                        color="red",
                        marker="s",
                        s=50,
                        label="Agreement",
                        zorder=4
                    )

        if plot_utils:
            for ax in fig_util.get_axes():
                ax.xaxis.set_minor_locator(tick.MultipleLocator(0.05))
                ax.yaxis.set_minor_locator(tick.MultipleLocator(0.05))
                ax.set_xlim(0, 1.05)
                ax.set_ylim(0, 1.05)
                ax.grid(color='gray', which='both', alpha=0.1, linestyle='--')
            fig_util.tight_layout()
            if path is not None:
                fig_util.savefig(path)
            else:
                fig_util.show()
        if plot_outcomes:
            for ax in fig_outcome.get_axes():
                ax.xaxis.set_minor_locator(tick.MultipleLocator(0.05))
                ax.yaxis.set_minor_locator(tick.MultipleLocator(0.05))
                ax.set_xlim(0, 1.05)
                ax.set_ylim(0, 1.05)
                ax.grid(color='gray', which='both', alpha=0.1, linestyle='--')
            fig_outcome.tight_layout()
            if path is not None:
                fig_outcome.savefig(path)
            else:
                fig_outcome.show()
