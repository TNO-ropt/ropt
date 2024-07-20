from typing import TYPE_CHECKING

import numpy as np
import pytest
from numpy.random import default_rng

from ropt.config.enopt import EnOptConfig
from ropt.ensemble_evaluator._gradient import _apply_bounds, _perturb_variables
from ropt.enums import BoundaryType
from ropt.plugins.sampler.scipy import SciPySampler

if TYPE_CHECKING:
    from numpy.typing import NDArray


def test__apply_bounds() -> None:
    variables: NDArray[np.float64] = np.array([-0.1, 0.2, 0.3, 0.4, 1.2])
    lower_bounds: NDArray[np.float64] = np.zeros(5, dtype=np.float64)
    upper_bounds: NDArray[np.float64] = np.ones(5, dtype=np.float64)

    # TRUNCATE BOTH
    expected_response_truncate_both: NDArray[np.float64] = np.array(
        [0.0, 0.2, 0.3, 0.4, 1.0],
    )
    response = _apply_bounds(
        variables, lower_bounds, upper_bounds, np.array(BoundaryType.TRUNCATE_BOTH)
    )
    assert expected_response_truncate_both == pytest.approx(response)

    # MIRROR_BOTH
    expected_response_mirror_both: NDArray[np.float64] = np.array(
        [0.1, 0.2, 0.3, 0.4, 0.8],
    )
    response = _apply_bounds(
        variables, lower_bounds, upper_bounds, np.array(BoundaryType.MIRROR_BOTH)
    )
    assert expected_response_mirror_both == pytest.approx(response)

    # MIRRORING STILL FAILS BOUNDS:
    variables = np.array([-1.1, 0.2, 0.3, 0.4, 2.2])
    expected_response_mirror_both = np.array([0.9, 0.2, 0.3, 0.4, 0.2])
    response = _apply_bounds(
        variables, lower_bounds, upper_bounds, np.array(BoundaryType.MIRROR_BOTH)
    )
    assert expected_response_mirror_both == pytest.approx(response)

    # MIRRORING STILL FAILS BOUNDS:
    variables = np.array([-100, 0.2, 0.3, 0.4, 100])
    expected_response_mirror_both = np.array([0.0, 0.2, 0.3, 0.4, 1.0])
    response = _apply_bounds(
        variables, lower_bounds, upper_bounds, np.array(BoundaryType.MIRROR_BOTH)
    )
    assert expected_response_mirror_both == pytest.approx(response)


def test_variable_perturbation_enopt() -> None:
    expected_perturbations = [
        [
            [
                0.013143932482607458,
                0.044310667426605846,
                0.1271962630644624,
                0.07259872095663066,
                0.10901154498199285,
                0.09195518956286257,
                0.03137681768145098,
                0.09039761102051466,
                0.045870227441709197,
                0.04568054419205199,
                0.06675836593352286,
                0.014296520325947581,
                0.12170830520508293,
                0.028645516241294522,
                0.11231347098297302,
                0.06921605619265588,
            ],
            [
                0.13920165398143983,
                0.029701529310408967,
                0.047210257176504115,
                0.07978845632794129,
                0.04737355490999022,
                0.10449607207793685,
                0.1402815197345309,
                0.11964033966325141,
                0.09943848221561255,
                0.054501105344238805,
                0.12599511135298563,
                0.11570153109859718,
                0.08173104222886356,
                0.06245571564144914,
                0.044211656391953846,
                0.0008883902254777504,
            ],
            [
                0.12391146464105754,
                0.045902194334259086,
                0.044292632707384226,
                0.0711190034837334,
                0.10599405923063684,
                0.15118306182559055,
                0.11286618879759056,
                0.04872392869507806,
                0.09810637789445714,
                0.0012800158459848124,
                0.13339692633443534,
                0.05417623375852973,
                0.028412042500647126,
                0.030246994978389737,
                0.05923193361898135,
                0.042760753871502834,
            ],
            [
                0.17709549736572894,
                0.02679094260597021,
                0.06443038715784852,
                0.06430249477928195,
                0.06441360613698689,
                0.06586729309763545,
                0.039121857090502546,
                0.034129624976794046,
                0.018591974896435784,
                0.01260873066257951,
                0.0727153416554606,
                0.11121881753479382,
                0.034945792949368594,
                0.03426924804781077,
                0.046475858533867626,
                0.03936801293805477,
            ],
            [
                0.009213487451247213,
                0.13095540160184596,
                0.08474999437787333,
                0.027315248640098377,
                0.07785858807699257,
                0.0411771363692134,
                0.052618128331266,
                0.08149819157895571,
                0.10934822459872673,
                0.1377762064620639,
                0.1470954724474519,
                0.049557064628452864,
                0.03711257044332452,
                0.06717987509610089,
                0.06871716061441564,
                0.02568890537775701,
            ],
            [
                0.03157623558826185,
                0.10336368602104241,
                0.14489005068703803,
                0.051574957581041386,
                0.030601739450233034,
                0.04893143966879051,
                0.0134434320014978,
                0.04965641151053459,
                0.08282220707338772,
                0.05773285192053224,
                0.1236288798530803,
                0.0695444016110217,
                0.0861559763667112,
                0.05476212337952535,
                0.12808328532801777,
                0.0012827198896714215,
            ],
            [
                0.04742043298498853,
                0.004015566215990432,
                0.1041136753505844,
                0.10541611448115122,
                0.037211620349336644,
                0.14600566591517442,
                0.048336870218037294,
                0.00586885599741864,
                0.04763977475819112,
                0.07980102549705377,
                0.05018039925364652,
                0.17527602999336045,
                0.10486933253899702,
                0.14876397193614144,
                0.13159427215454714,
                0.021889233918437848,
            ],
            [
                0.043712069666742515,
                0.07372428434364228,
                0.030480163204652022,
                0.11865521278918569,
                0.020839445203028613,
                0.031265578101691074,
                0.07950670837631815,
                0.10266582505951691,
                0.04388433079339729,
                0.04900939122799271,
                0.10234908512513213,
                0.07470293424320865,
                0.04783566144430481,
                0.07792517461274441,
                0.06610607540751777,
                0.07513164617494864,
            ],
            [
                0.16266155909013308,
                0.04743960381623062,
                0.03581185710267708,
                0.133581539705488,
                0.0277149677960705,
                0.14909447378386242,
                0.0534901096894021,
                0.06699197590221344,
                0.09838141405704356,
                0.12245548477065502,
                0.1117499325833947,
                0.05181823223721199,
                0.04377855822510775,
                0.07957363230274925,
                0.06903595361156795,
                0.030286344638481726,
            ],
            [
                0.08313640374202519,
                0.11240598100807374,
                0.07112533524869327,
                0.14109998585254874,
                0.08351510646372082,
                0.05530934995968665,
                0.06821074449657094,
                0.035294506641587085,
                0.010135101593650324,
                0.08562642556665054,
                0.01923748412195367,
                0.04408540138350989,
                0.057639997314752556,
                0.0814552087091204,
                0.0853580993305465,
                0.06511117714138812,
            ],
        ],
    ]
    variables = [
        0.0626,
        0.0627,
        0.0628,
        0.0629,
        0.0630,
        0.0631,
        0.0632,
        0.0633,
        0.0617,
        0.0618,
        0.0619,
        0.0620,
        0.0621,
        0.0622,
        0.0623,
        0.0624,
    ]

    config_dict = {
        "variables": {
            "initial_values": [0.0] * len(variables),
            "lower_bounds": 0.0,
        },
        "optimizer": {"method": "opt"},
        "objective_functions": {"weights": [1.0]},
        "gradient": {"number_of_perturbations": 10, "perturbation_magnitudes": 0.05},
    }

    config = EnOptConfig.model_validate(config_dict)
    sampler = SciPySampler(config, 0, None, default_rng(123))
    perturbations = _perturb_variables(
        np.array(variables), config.variables, config.gradient, [sampler]
    )
    assert expected_perturbations == pytest.approx(perturbations)
