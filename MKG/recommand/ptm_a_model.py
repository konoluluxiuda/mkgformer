import numpy as np
import time


class PTMASchemeA:
    """
    PTM(a) scheme-A adapter for NEWHERB:
    - Treat disease as single symptom token.
    - Treat train herbs for each disease as one prescription's herb bag.
    - Learn topic-role assignments with collapsed Gibbs sampling.
    """

    def __init__(
        self,
        num_topics=20,
        num_roles=4,
        alpha=1.0,
        beta=0.1,
        beta_bar=0.1,
        eta=1.0,
        iterations=200,
        log_interval=1,
        seed=42,
    ):
        self.K = int(num_topics)
        self.X = int(num_roles)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.beta_bar = float(beta_bar)
        self.eta = float(eta)
        self.iterations = int(iterations)
        self.log_interval = max(1, int(log_interval))
        self.rng = np.random.RandomState(seed)

        self._fitted = False

    def fit(
        self,
        train_dict,
        herb_indices,
        disease_ids,
        max_iterations=None,
        on_iteration_end=None,
    ):
        self.herb_indices = list(herb_indices)
        self.disease_ids = list(disease_ids)
        self.H = len(self.herb_indices)
        self.S = len(self.disease_ids)

        self.herb_to_local = {h: i for i, h in enumerate(self.herb_indices)}
        self.disease_to_local = {d: i for i, d in enumerate(self.disease_ids)}
        self.disease_to_doc = {}

        herbs_docs = []
        symptoms_docs = []

        for d_local, d_global in enumerate(self.disease_ids):
            pos_herbs = [h for h in train_dict.get(d_global, []) if h in self.herb_to_local]
            if len(pos_herbs) == 0:
                continue

            p = len(herbs_docs)
            self.disease_to_doc[d_global] = p

            herb_local_ids = np.asarray([self.herb_to_local[h] for h in pos_herbs], dtype=np.int64)
            herbs_docs.append(herb_local_ids)
            symptoms_docs.append(np.asarray([d_local], dtype=np.int64))

        self.herbs_docs = herbs_docs
        self.symptoms_docs = symptoms_docs
        self.P = len(self.herbs_docs)

        if self.P == 0:
            raise ValueError("No valid training prescriptions for PTM scheme-A.")

        self._init_state()

        total_iters = int(max_iterations) if max_iterations is not None else self.iterations
        train_start = time.time()
        for it in range(total_iters):
            self._gibbs_once()
            if ((it + 1) % self.log_interval) == 0:
                elapsed = time.time() - train_start
                avg = elapsed / (it + 1)
                remain = avg * (total_iters - it - 1)
                print(
                    f"[PTM-A] Iter {it+1}/{total_iters} | "
                    f"elapsed={elapsed:.1f}s | eta={remain:.1f}s"
                )

            if on_iteration_end is not None:
                should_stop = on_iteration_end(self, it + 1, total_iters)
                if should_stop:
                    print(f"[PTM-A] Early stop at iter {it+1}/{total_iters}")
                    break

        self.refresh_posterior()

    def refresh_posterior(self):
        self.theta = self._estimate_theta()
        self.phi = self._estimate_phi()
        self.phi_bar = self._estimate_phi_bar()
        self.pi = self._estimate_pi()
        self._fitted = True

    def get_snapshot(self):
        if not self._fitted:
            self.refresh_posterior()
        return {
            'theta': self.theta.copy(),
            'phi': self.phi.copy(),
            'phi_bar': self.phi_bar.copy(),
            'pi': self.pi.copy(),
        }

    def load_snapshot(self, snapshot):
        self.theta = snapshot['theta']
        self.phi = snapshot['phi']
        self.phi_bar = snapshot['phi_bar']
        self.pi = snapshot['pi']
        self._fitted = True

    def score_disease(self, disease_global_id):
        if not self._fitted:
            raise RuntimeError("Model is not fitted.")

        if disease_global_id not in self.disease_to_doc:
            return np.zeros(self.H, dtype=np.float64)

        p = self.disease_to_doc[disease_global_id]
        s_local = self.disease_to_local[disease_global_id]

        # Fuse prescription-topic and symptom-topic signals.
        w_k = self.theta[p] * self.phi_bar[:, s_local]
        w_sum = w_k.sum()
        if w_sum <= 0:
            w_k = self.theta[p].copy()
            w_sum = w_k.sum()
        w_k = w_k / (w_sum + 1e-12)

        # score[h] = sum_k w_k * sum_r pi[p,k,r] * phi[k,r,h]
        role_mix = (self.pi[p][:, :, None] * self.phi).sum(axis=1)  # [K, H]
        score = (w_k[:, None] * role_mix).sum(axis=0)  # [H]
        return score

    def _init_state(self):
        K, X, H, S, P = self.K, self.X, self.H, self.S, self.P

        self.np_count = np.zeros((P, K), dtype=np.int64)
        self.npsum = np.zeros(P, dtype=np.int64)

        self.ns = np.zeros((S, K), dtype=np.int64)
        self.nssum = np.zeros(K, dtype=np.int64)

        self.kxh = np.zeros((K, X, H), dtype=np.int64)
        self.nhsum = np.zeros((K, X), dtype=np.int64)

        self.nx = np.zeros((P, K, X), dtype=np.int64)
        self.nxsum = np.zeros((P, K), dtype=np.int64)

        self.treatment = []
        self.roles = []
        self.syndrome = []

        for p in range(P):
            herbs = self.herbs_docs[p]
            syms = self.symptoms_docs[p]

            tp = np.zeros(len(herbs), dtype=np.int64)
            rp = np.zeros(len(herbs), dtype=np.int64)

            for n, h_local in enumerate(herbs):
                t = self.rng.randint(0, K)
                r = self.rng.randint(0, X)
                tp[n] = t
                rp[n] = r
                self._update_count_herb(p, t, r, h_local, +1)

            sp = np.zeros(len(syms), dtype=np.int64)
            for m, s_local in enumerate(syms):
                syn = self.rng.randint(0, K)
                sp[m] = syn
                self._update_count_symptom(p, syn, s_local, +1)

            self.treatment.append(tp)
            self.roles.append(rp)
            self.syndrome.append(sp)

    def _gibbs_once(self):
        for p in range(self.P):
            herbs = self.herbs_docs[p]
            for n, h_local in enumerate(herbs):
                new_t, new_r = self._sample_treat_role(p, n, h_local)
                self.treatment[p][n] = new_t
                self.roles[p][n] = new_r

            syms = self.symptoms_docs[p]
            for m, s_local in enumerate(syms):
                new_syn = self._sample_syndrome(p, m, s_local)
                self.syndrome[p][m] = new_syn

    def _sample_treat_role(self, p, n, h_local):
        t_old = self.treatment[p][n]
        r_old = self.roles[p][n]
        self._update_count_herb(p, t_old, r_old, h_local, -1)

        left = (self.np_count[p] + self.alpha) / (self.npsum[p] + self.K * self.alpha)  # [K]
        role_prob = (self.nx[p] + self.eta) / (self.nxsum[p][:, None] + self.X * self.eta)  # [K, X]
        herb_prob = (self.kxh[:, :, h_local] + self.beta) / (self.nhsum + self.H * self.beta)  # [K, X]
        pr = left[:, None] * role_prob * herb_prob  # [K, X]

        flat = pr.reshape(-1)
        flat_sum = flat.sum()
        if flat_sum <= 0:
            idx = self.rng.randint(0, flat.size)
        else:
            flat = flat / flat_sum
            idx = self.rng.choice(flat.size, p=flat)

        t_new = idx // self.X
        r_new = idx % self.X
        self._update_count_herb(p, t_new, r_new, h_local, +1)
        return t_new, r_new

    def _sample_syndrome(self, p, m, s_local):
        syn_old = self.syndrome[p][m]
        self._update_count_symptom(p, syn_old, s_local, -1)

        pr = (
            (self.np_count[p] + self.alpha) / (self.npsum[p] + self.K * self.alpha)
            * (self.ns[s_local] + self.beta_bar) / (self.nssum + self.S * self.beta_bar)
        )

        pr_sum = pr.sum()
        if pr_sum <= 0:
            syn_new = self.rng.randint(0, self.K)
        else:
            pr = pr / pr_sum
            syn_new = self.rng.choice(self.K, p=pr)

        self._update_count_symptom(p, syn_new, s_local, +1)
        return syn_new

    def _update_count_symptom(self, p, syn, s_local, flag):
        self.ns[s_local, syn] += flag
        self.nssum[syn] += flag

        self.np_count[p, syn] += flag
        self.npsum[p] += flag

    def _update_count_herb(self, p, t, r, h_local, flag):
        self.np_count[p, t] += flag
        self.npsum[p] += flag

        self.kxh[t, r, h_local] += flag
        self.nhsum[t, r] += flag

        self.nx[p, t, r] += flag
        self.nxsum[p, t] += flag

    def _estimate_theta(self):
        return (self.np_count + self.alpha) / (self.npsum[:, None] + self.K * self.alpha)

    def _estimate_phi_bar(self):
        # [K, S]
        return (self.ns.T + self.beta_bar) / (self.nssum[:, None] + self.S * self.beta_bar)

    def _estimate_phi(self):
        # [K, X, H]
        den = self.nhsum[:, :, None] + self.H * self.beta
        return (self.kxh + self.beta) / den

    def _estimate_pi(self):
        # [P, K, X]
        den = self.nxsum[:, :, None] + self.X * self.eta
        return (self.nx + self.eta) / den
