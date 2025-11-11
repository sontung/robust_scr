import re

import numpy as np


aachen = {
    "hloc": "89.6&95.4&98.8&86.7&93.9&100.0",
    "as": "85.3&92.2&97.9&39.8&49.0&64.3",
    "ace": "6.9&17.2&50.0&0.0&1.0&5.1",
    "deviloc": "87.4 / 94.8 / 98.2 87.8 / 93.9 / 100.",
    "cascaded": "76.7&88.6&95.8&33.7&48.0&62.2",
    "squeezer": "75.5&89.7&96.2&50.0&67.3&78.6",
    "hscnet": "72.7&81.6&91.4&43.9&57.1&76.5",
    "esac": "42.6 59.6 75.5 6.1 10.2 18.4",
    "neumap": "80.8&90.9&95.6&48.0&67.3&87.8",
    "rscore": "79.0&88.5&96.4&66.3&89.8&96.9",
    "pixloc": "64.3 69.3 77.4 51.1 55.1 67.3",
    "glace": "8.6 / 20.8 / 64.0	1.0 / 1.0 / 17.3",
    "us_netvlad": "			79.9 / 91.0 / 97.2	73.5 / 90.8 / 96.9",
    "us_vani": "			72.3 / 85.8 / 95.4	58.2 / 80.6 / 94.9",
    "us+sampling": "		77.9 / 89.6 / 96.4	68.4 / 88.8 / 96.9",
    "us_+salad": "			78.2 / 89.7 / 96.4	69.4 / 90.8 / 99.0",
    "us_salad": "	80.3 / 90.3 / 97.1	72.4 / 90.8 / 99.0",
    "us_megaloc": "79.4 / 90.8 / 96.8	73.5 / 91.8 / 99.0",
    "us_eigenplaces": "	79.5 / 90.3 / 96.0	70.4 / 91.8 / 96.9",
    "us_mixvpr": "	79.9 / 90.8 / 97.2	73.5 / 89.8 / 96.9",
    "us_boq": "	79.7 / 90.5 / 96.8	74.5 / 91.8 / 99.0",
    "2layers": "80.6 / 91.3 / 97.1	70.4 / 93.9 / 99.0",
    "4layers": "	79.9 / 90.9 / 97.2	70.4 / 91.8 / 99.0",
}

hyundai = {
    "hloc": "80.6&84.3&89.4&85.3&91.0&93.1&75.2&80.3&87.6",
    "hloc d2": "78.0 82.8 88.0 84.2 89.8 92.0 73.7 79.3 87.2",
    "deviloc": "	86.9 / 91.5 / 96.3	88.7 / 93.7 / 96.1	78.5 / 84.2 / 93.7",
    "ace": "14.1 & 54.4 & 75.5  &27.3&70.9&84.1&2.7&14.4&29.3",
    "esac": "43.3 66.3 77.0 45.2 62.5 73.1  3.5 8.2 12.6",
    "glace": "5.6 & 21.3 & 48.6  &8.4&29.8&51.6&0.9&4.4&11.9",
    "rscore": "63.9&83.3&90.8&76.7&89.3&93.0&61.5&77.6&88.8",
    "rscore_loftr": "67.3 / 84.5 / 92.6 70.5 / 87.0 / 92.9 30.8 / 53.7 / 72.7",
    "us": "70.4 / 85.6 / 92.0	77.6 / 88.8 / 92.1	66.6 / 81.7 / 92.6",
    "us2": "	70.6 / 85.1 / 92.4	79.0 / 89.7 / 93.5	64.7 / 79.6 / 90.4",
}

aachen11 = {
    "hloc": "89.8&96.1&99.4 & 77.0&90.6&100.0",
    "rscore": "76.8 / 90.0 / 97.6	53.4 / 82.2 / 97.9",
    "us": "82.8 / 93.1 / 98.8	53.4 / 84.8 / 99.0",
}


def find_numbers(string_, return_numbers=False):
    pattern = r"[-+]?(?:\d*\.*\d+)"
    matches = re.findall(pattern, string_)
    numbers = list(map(float, matches))
    if return_numbers:
        return numbers
    avg = sum(numbers) / len(matches)
    return avg


method_dict = {}
for name, ds in [["aachen", aachen], ["hyundai", hyundai], ["aachen11", aachen11]]:
    print(name)
    for method_ in ds:
        count = np.array([0, 0, 0])
        scores = np.array([0, 0, 0], dtype=float)
        res = ds[method_]
        if len(res) == 0:
            continue
        if "&" not in res:
            res = find_numbers(res, True)
        else:
            res = list(map(float, res.split("&")))
        avg = np.round(np.average(res), 1)
        res.append(avg)
        print(method_, "&".join([f"{{{x}}}" for x in res]))
    print()
