"""
parse_demo.py
Парсить CS2 .dem файл через awpy та повертає структуровані дані:
раунди, тіки (позиції/здоров'я/зброя), вбивства, гранати.
"""
from awpy import Demo
import polars as pl


def parse_demo(path: str) -> dict:
    """Парсить demo файл і повертає словник з ключовими датафреймами."""
    dem = Demo(path)
    dem.parse(
        player_props=[
            "X", "Y", "Z",
            "pitch", "yaw",
            "health",
            "armor_value",
            "has_helmet",
            "has_defuser",
            "inventory",
            "current_equip_value",
            "team_name",
        ]
    )

    return {
        "header": dem.header,
        "rounds": dem.rounds,      # межі раундів (старт/кінець тіків), переможець
        "ticks": dem.ticks,        # позиції/здоров'я по кожному гравцю на кожному тіку
        "kills": dem.kills,        # усі вбивства
        "damages": dem.damages,    # уся завдана шкода
        "grenades": dem.grenades,  # кидки гранат
        "bomb": dem.bomb,          # плант/дефьюз бомби
    }


def summarize(parsed: dict) -> None:
    """Швидкий друк базової інформації про демку — для перевірки, що парсинг пройшов ок."""
    header = parsed["header"]
    rounds = parsed["rounds"]
    print(f"Мапа: {header.get('map_name', '?')}")
    print(f"Раундів: {len(rounds)}")
    print(f"Тіків у ticks-датафреймі: {len(parsed['ticks'])}")
    print(f"Вбивств: {len(parsed['kills'])}")
    print(rounds.head(5))


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Використання: python parse_demo.py шлях_до_демки.dem")
        sys.exit(1)
    parsed = parse_demo(sys.argv[1])
    summarize(parsed)
