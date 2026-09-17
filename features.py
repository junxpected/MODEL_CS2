import polars as pl


def get_key_moments(parsed: dict) -> pl.DataFrame:
    kills = parsed["kills"]
    rounds = parsed["rounds"]
    ticks = parsed["ticks"]

    moments = []
    for rnd in rounds.iter_rows(named=True):
        round_num = rnd.get("round_num") or rnd.get("round")
        start_tick = rnd.get("official_start") or rnd.get("start")
        end_tick = rnd.get("official_end") or rnd.get("end")
        winner = rnd.get("winner")

        round_kills = kills.filter(
            (pl.col("tick") >= start_tick) & (pl.col("tick") <= end_tick)
        ).sort("tick")

        for kill in round_kills.iter_rows(named=True):
            tick = kill["tick"]
            tick_state = ticks.filter(pl.col("tick") == tick)
            if tick_state.height == 0:
                continue

            ct_alive = tick_state.filter(
                (pl.col("side") == "ct") & (pl.col("health") > 0)
            ).height
            t_alive = tick_state.filter(
                (pl.col("side") == "t") & (pl.col("health") > 0)
            ).height
            ct_equip = tick_state.filter(pl.col("side") == "ct")["current_equip_value"].sum()
            t_equip = tick_state.filter(pl.col("side") == "t")["current_equip_value"].sum()

            moments.append({
                "round_num": round_num,
                "tick": tick,
                "ct_alive": ct_alive,
                "t_alive": t_alive,
                "ct_equip_value": ct_equip,
                "t_equip_value": t_equip,
                "ct_health_sum": tick_state.filter(pl.col("side") == "ct")["health"].sum(),
                "t_health_sum": tick_state.filter(pl.col("side") == "t")["health"].sum(),
                "seconds_into_round": (tick - (rnd.get("freeze_end") or start_tick)) / 64.0,
                "attacker": kill.get("attacker_name"),
                "victim": kill.get("victim_name"),
                "attacker_side": kill.get("attacker_side"),
                "victim_side": kill.get("victim_side"),
                "round_winner": winner,
            })

    return pl.DataFrame(moments)


def sample_round_states(parsed: dict, sample_every_n_ticks: int = 128) -> pl.DataFrame:
    rounds = parsed["rounds"]
    ticks = parsed["ticks"]

    samples = []
    for rnd in rounds.iter_rows(named=True):
        round_num = rnd.get("round_num")
        start_tick = rnd.get("freeze_end") or rnd.get("start")
        end_tick = rnd.get("end")
        winner = rnd.get("winner")
        if start_tick is None or end_tick is None or winner is None:
            continue

        round_ticks = ticks.filter(
            (pl.col("round_num") == round_num) &
            (pl.col("tick") >= start_tick) &
            (pl.col("tick") <= end_tick)
        )
        if round_ticks.height == 0:
            continue

        unique_ticks = sorted(round_ticks["tick"].unique().to_list())
        sampled_ticks = unique_ticks[::sample_every_n_ticks]

        for tick in sampled_ticks:
            state = round_ticks.filter(pl.col("tick") == tick)
            ct = state.filter(pl.col("side") == "ct")
            t = state.filter(pl.col("side") == "t")
            ct_alive = ct.filter(pl.col("health") > 0).height
            t_alive = t.filter(pl.col("health") > 0).height

            samples.append({
                "round_num": round_num,
                "tick": tick,
                "seconds_into_round": (tick - start_tick) / 64.0,
                "ct_alive": ct_alive,
                "t_alive": t_alive,
                "ct_health_sum": ct["health"].sum(),
                "t_health_sum": t["health"].sum(),
                "ct_equip_value": ct["current_equip_value"].sum(),
                "t_equip_value": t["current_equip_value"].sum(),
                "ct_won": 1 if winner == "ct" else 0,
            })

    return pl.DataFrame(samples)
