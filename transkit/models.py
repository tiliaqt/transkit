import lmfit as lm
import numpy as np

from transkit import fit


##############
### Models ###


class PKModel:
    def fit(self, Ct: np.ndarray) -> fit.RefinedFit:
        return fit.minimize_refine(
            self.residuals,
            self.params,
            ["basinhopping"],
            residuals_args=(Ct,),
        )

    def residuals(self, params, Ct):
        return self.model_params(Ct[:, 0], params) - Ct[:, 1]


class Cmpt1Model(PKModel):
    def model(self, t, D, ka, ke, Vd):
        C = (D * ka) / (Vd * (ka - ke)) * (np.exp(-ke * t) - np.exp(-ka * t))
        return C

    def model_params(self, t, params):
        pvd = params.valuesdict()
        D = pvd["D"]
        ka = pvd["ka"]
        ke = pvd["ke"]
        Vd = pvd["Vd"]

        return self.model(t, D, ka, ke, Vd)

    def params(self, seed=None):
        D_val = 1e6 if seed is None or "D" not in seed else seed["D"].value
        ka_val = 1.0 if seed is None or "ka" not in seed else seed["ka"].value
        ke_val = 1.1 if seed is None or "ke" not in seed else seed["ke"].value
        Vd_val = 1.2 if seed is None or "Vd" not in seed else seed["Vd"].value

        params = lm.Parameters()
        params.add("D", value=D_val, min=1e-9, max=1e9)
        params.add("ka", value=ka_val, min=1e-6, max=1e6)
        params.add("ke", value=ke_val, min=1e-6, max=1e6)
        params.add("Vd", value=Vd_val, min=1e-6, max=1e6)

        return params


class Cmpt2Model(PKModel):
    def model(self, t, D, ka, k21, a, b, V):
        A = (ka / V) * ((k21 - a) / (ka - a) * (b - a))
        B = (ka / V) * ((k21 - b) / (ka - b) * (a - b))
        Cp = D * (
            A * np.exp(-a * t) + B * np.exp(-b * t) - (A + B) * np.exp(-ka * t)
        )
        return Cp

    def model_params(self, t, params):
        pvd = params.valuesdict()
        D = pvd["D"]
        ka = pvd["ka"]
        k21 = pvd["k21"]
        a = pvd["a"]
        b = pvd["b"]
        V = pvd["V"]

        return self.model(t, D, ka, k21, a, b, V)

    def params(self, seed=None):
        D_val = 1e8 if seed is None or "D" not in seed else seed["D"].value
        ka_val = 1.0 if seed is None or "ka" not in seed else seed["ka"].value
        k21_val = (
            1.1 if seed is None or "k21" not in seed else seed["k21"].value
        )
        a_val = 1.2 if seed is None or "a" not in seed else seed["a"].value
        b_val = 1.3 if seed is None or "b" not in seed else seed["b"].value
        V_val = 1.4 if seed is None or "V" not in seed else seed["V"].value

        params = lm.Parameters()
        params.add("D", value=D_val, min=1e-9, max=1e9)
        params.add("ka", value=ka_val, min=1e-6, max=1e6)
        params.add("k21", value=k21_val, min=1e-6, max=1e6)
        params.add("a", value=a_val, min=1e-6, max=1e6)
        params.add("b", value=b_val, min=1e-6, max=1e6)
        params.add("V", value=V_val, min=1e-6, max=1e6)

        return params


class Cmpt3Model(PKModel):
    def model(t, D, ka, k21, k31, a, b, g, V):
        A = (
            (1 / V)
            * (ka / (ka - a))
            * ((k21 - a) / (a - b))
            * ((k31 - a) / (a - g))
        )
        B = (
            (1 / V)
            * (ka / (ka - b))
            * ((k21 - b) / (b - a))
            * ((k31 - b) / (b - g))
        )
        C = (
            (1 / V)
            * (ka / (ka - g))
            * ((k21 - g) / (g - b))
            * ((k31 - g) / (g - a))
        )
        Cp = D * (
            A * np.exp(-a * t)
            + B * np.exp(-b * t)
            + C * np.exp(-g * t)
            - ((A + B + C) * np.exp(-ka * t))
        )
        return Cp

    def model_params(self, t, params):
        pvd = params.valuesdict()
        D = pvd["D"]
        ka = pvd["ka"]
        k21 = pvd["k21"]
        k31 = pvd["k31"]
        a = pvd["a"]
        b = pvd["b"]
        g = pvd["g"]
        V = pvd["V"]

        return self.model(t, D, ka, k21, k31, a, b, g, V)

    def params(self, seed=None):
        D_val = 1e8 if seed is None or "D" not in seed else seed["D"].value
        ka_val = 1.0 if seed is None or "ka" not in seed else seed["ka"].value
        k21_val = (
            1.1 if seed is None or "k21" not in seed else seed["k21"].value
        )
        k31_val = (
            1.2 if seed is None or "k31" not in seed else seed["k31"].value
        )
        a_val = 1.3 if seed is None or "a" not in seed else seed["a"].value
        b_val = 1.4 if seed is None or "b" not in seed else seed["b"].value
        g_val = 1.5 if seed is None or "g" not in seed else seed["g"].value
        V_val = 1.6 if seed is None or "V" not in seed else seed["V"].value

        params = lm.Parameters()
        params.add("D", value=D_val, min=1e-9, max=1e9)
        params.add("ka", value=ka_val, min=1e-6, max=1e3)
        params.add("k21", value=k21_val, min=1e-6, max=1e3)
        params.add("k31", value=k31_val, min=1e-6, max=1e3)
        params.add("a", value=a_val, min=1e-6, max=1e6)
        params.add("b", value=b_val, min=1e-6, max=1e6)
        params.add("g", value=g_val, min=1e-6, max=1e6)
        params.add("V", value=V_val, min=1e-6, max=1e6)

        return params


def _cmpt3v(t, D, k1, k2, k3):
    k1t = np.clip(-k1 * t, None, 300.0)
    k2t = np.clip(-k2 * t, None, 300.0)
    k3t = np.clip(-k3 * t, None, 300.0)

    C_num = (
        D
        * k1
        * k2
        * (
            (k2 - k3) * np.exp(k1t)
            + (k3 - k1) * np.exp(k2t)
            + (k1 - k2) * np.exp(k3t)
        )
    )
    C_den = (k1 - k2) * (k1 - k3) * (k2 - k3)

    if C_den != 0:
        return C_num / C_den
    else:
        return 1e20


class Cmpt3VModel(PKModel):
    def model(self, t, D, k1, k2, k3):
        return _cmpt3v(t, D, k1, k2, k3)

    def model_params(self, t, params):
        pvd = params.valuesdict()
        D = pvd["D"]
        k1 = pvd["k1"]
        k2 = pvd["k2"]
        k3 = pvd["k3"]
        return self.model(t, D, k1, k2, k3)

    def params(self, seed=None):
        D_val = 1e6 if seed is None or "D" not in seed else seed["D"].value
        k1_val = 0.1 if seed is None or "k1" not in seed else seed["k1"].value
        k2_val = 0.2 if seed is None or "k2" not in seed else seed["k2"].value
        k3_val = 0.3 if seed is None or "k3" not in seed else seed["k3"].value

        params = lm.Parameters()
        params.add("D", value=D_val, min=1e-9, max=1e9)
        params.add("k1", value=k1_val, min=1e-6, max=1e3)
        params.add("k2", value=k2_val, min=1e-6, max=1e3)
        params.add("k3", value=k3_val, min=1e-6, max=1e3)

        return params


def _cmpt4v(t, D, k1, k2, k3, k4):
    k1t = np.clip(-k1 * t, None, 500.0)
    k2t = np.clip(-k2 * t, None, 500.0)
    k3t = np.clip(-k3 * t, None, 500.0)
    k4t = np.clip(-k4 * t, None, 500.0)

    C_num = (
        -D
        * k1
        * k2
        * k3
        * (
            (
                k2 * k2 * k3
                - k2 * k3 * k3
                - k2 * k2 * k4
                + k3 * k3 * k4
                + k2 * k4 * k4
                - k3 * k4 * k4
            )
            * np.exp(k1t)
            + (
                -k1 * k1 * k3
                + k1 * k3 * k3
                + k1 * k1 * k4
                - k3 * k3 * k4
                - k1 * k4 * k4
                + k3 * k4 * k4
            )
            * np.exp(k2t)
            + (
                k1 * k1 * k2
                - k1 * k2 * k2
                - k1 * k1 * k4
                + k2 * k2 * k4
                + k1 * k4 * k4
                - k2 * k4 * k4
            )
            * np.exp(k3t)
            + (
                -k1 * k1 * k2
                + k1 * k2 * k2
                + k1 * k1 * k3
                - k2 * k2 * k3
                - k1 * k3 * k3
                + k2 * k3 * k3
            )
            * np.exp(k4t)
        )
    )
    C_den = (
        (k1 - k2) * (k1 - k3) * (k2 - k3) * (k1 - k4) * (k2 - k4) * (k3 - k4)
    )

    if C_den != 0:
        return C_num / C_den
    else:
        return np.full_like(t, 1e20)


class Cmpt4VModel(PKModel):
    def model(self, t, D, k1, k2, k3, k4):
        return _cmpt4v(t, D, k1, k2, k3, k4)

    def model_params(self, t, params):
        pvd = params.valuesdict()
        D = pvd["D"]
        k1 = pvd["k1"]
        k2 = pvd["k2"]
        k3 = pvd["k3"]
        k4 = pvd["k4"]

        return self.model(t, D, k1, k2, k3, k4)

    def params(self, seed=None):
        D_val = 1e6 if seed is None or "D" not in seed else seed["D"].value
        k1_val = 0.1 if seed is None or "k1" not in seed else seed["k1"].value
        k2_val = 0.2 if seed is None or "k2" not in seed else seed["k2"].value
        k3_val = 0.3 if seed is None or "k3" not in seed else seed["k3"].value
        k4_val = 0.4 if seed is None or "k4" not in seed else seed["k4"].value

        params = lm.Parameters()
        params.add("D", value=D_val, min=1e-9, max=1e9)
        params.add("k1", value=k1_val, min=1e-6, max=1e3)
        params.add("k2", value=k2_val, min=1e-6, max=1e3)
        params.add("k3", value=k3_val, min=1e-6, max=1e3)
        params.add("k4", value=k4_val, min=1e-6, max=1e3)

        return params


class PolyExpModel(PKModel):
    def coeffs(self, params):
        pvd = params.valuesdict()

        coeffs = []
        for p, v in sorted(pvd.items(), key=lambda it: it[0][1] + it[0][0]):
            if p[0] == "A" or p[0] == "B" or p[0] == "L":
                coeffs.append(v)

        return np.array(coeffs)

    def fit(self, Ct: np.ndarray, n: int) -> fit.RefinedFit:
        return fit.minimize_refine(
            self.residuals,
            self.params,
            ["basinhopping"],
            residuals_args=(Ct,),
            params_args=(n,),
        )

    def model(self, t, coeffs):
        cf = coeffs.reshape(int(coeffs.size / 2), 2)
        exp_term = np.clip(-cf[:, 1:2] * t, None, 500.0)
        ret = np.sum(cf[:, 0:1] * np.exp(exp_term), axis=0)
        return ret

    def model_params(self, t, params):
        return self.model(t, self.coeffs(params))

    def residuals(self, params, Ct):
        coeffs = self.coeffs(params)
        model = self.model(Ct[:, 0], coeffs)
        res = model - Ct[:, 1]

        # Try to reject solutions where any of the exponents are equal,
        # or where the model is negative.
        # TODO: interpolate the model, don't just test for < 0 a t points
        # in Ct. Have to interpolate to avoid over-fitting (but doing that
        # interpolation should resolve the over-fitting issues!)
        eq_coeffs = coeffs[1::2, None] == coeffs[None, 1::2]
        np.fill_diagonal(eq_coeffs, False)
        if np.any(eq_coeffs) or np.any(model < 0.0):
            res[:] = 1e40

        return res

    def params(self, n, seed=None):
        params = lm.Parameters()

        for i in range(0, n):
            Ai = f"A{i+1}"
            Bi = f"B{i+1}"
            Ai_v = None
            Bi_v = None

            if seed is not None:
                if isinstance(seed, lm.Parameters):
                    if Ai in seed:
                        Ai_v = seed[Ai].value
                    if Bi in seed:
                        Bi_v = seed[Bi].value
                elif isinstance(seed, np.ndarray):
                    Ai_v = seed[2 * i]
                    Bi_v = seed[2 * i + 1]
                else:
                    raise ValueError(
                        "seed can only be of type lmfit.Parameters or "
                        f"numpy.ndarray, but got {type(seed)}."
                    )

            Ai_v = 1e4 if Ai_v is None else Ai_v
            Bi_v = 0.1 * (i + 1) if Bi_v is None else Bi_v

            if i == n - 1 and n > 1:
                # The sum of all A_i must equal 0 to ensure the fitted function
                # goes through the origin.
                sum_expr = "".join([f"-A{j+1}" for j in range(0, i)])
            else:
                sum_expr = None

            params.add(Bi, value=Bi_v, min=1e-6, max=1e2)
            params.add(Ai, value=Ai_v, expr=sum_expr)

        return params
