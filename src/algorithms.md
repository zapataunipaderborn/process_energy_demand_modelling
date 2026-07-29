\begin{algorithm}
\caption{Extraction of the process model.}
\label{alg:pm_extraction_ml}
\begin{algorithmic}[1]
\Require Event log $L$ over cases $C$ and activities $A$, each event carrying its object, attributes $\mathbf{x}^{\mathrm{attr}}$ and external features $\mathbf{x}^{\mathrm{ef}}$ via $\pi_{data}(e)$; process miners $\mathcal{M}$ with configuration spaces $\Theta_m$; parametric duration families $\mathcal{P}$; regressors $\mathcal{R}$; minimum sample size $n_{\min}$

\State \textbf{(A) Process Table (PT)}
\State Split every activity with nested sub-activities until each one has a characteristic energy profile, and assemble $PT$ with one row per event: $(\pi_{case}(e),\; \pi_{act}(e),\; \pi_{time}(e),\; \mathbf{x}^{\mathrm{attr}},\; \mathbf{x}^{\mathrm{ef}})$

\State \textbf{(B) Petri net structure} $(N, M_0, M_f)$
\For{each sub-log induced by an object and its higher-level activity}
    \State For every miner $m \in \mathcal{M}$ and configuration $\theta \in \Theta_m$, mine the Petri net $(N_{m,\theta},\, M_{0,m,\theta},\, M_{f,m,\theta})$ with labelling $\ell : T \rightarrow A \cup \{\tau\}$, and score $q(m,\theta)$ as the mean of replay fitness, precision, generalisation and simplicity
    \State Set $(N, M_0, M_f, \ell)$ to the Petri net maximising $q$
\EndFor

\State \textbf{(C) Firing-time distributions} $\mathcal{G}$
\For{each activity $a \in A$ with $n_a$ observed durations $d$}
    \State Fit every family in $\mathcal{P}$ to $d$ by maximum likelihood estimation and keep as $\hat{p}_a$ the one with the highest Kolmogorov--Smirnov $p$-value
    \If{$n_a \geq n_{\min}$}
        \State Build $X_a$ from $\mathbf{x}^{\mathrm{attr}}$, $\mathbf{x}^{\mathrm{ef}}$, the position of $a$ in the trace, the elapsed case time and the lagged durations
        \State Fit each $h \in \mathcal{R}$ on $d$ by $K$-fold cross-validation, keep $\hat{f}_a = \arg\min_{h} \mathrm{MAE}_{\mathrm{CV}}(h)$ and rescale it by its out-of-fold ratio $\bar{d} / \bar{\hat{d}}^{\mathrm{oof}}$, so that it predicts the mean of the right-skewed durations
    \EndIf
\EndFor
\State Fit the global model $\hat{f}_{\mathrm{global}}$ likewise on all activities pooled, with the median duration of the activity as an additional feature
\State Set $\mathcal{G} = \{D_t\}_{t \in T}$, with $D_t$ immediate for every $\tau$-transition and, for every visible transition, either sampled from $\hat{p}_{\ell(t)}$ or predicted by $\hat{f}_{\ell(t)}$ or $\hat{f}_{\mathrm{global}}$

\State \textbf{(D) Routing weights} $w$\textbf{, case behaviour and duration budget}
\State Set the routing weight $w(t)$ of every transition to the number of times it fires in the token replay of the training log, and $w(t) = 1$ for transitions that are never replayed, so that unobserved behaviour keeps a small residual probability
\State For every activity $a$, record the empirical distribution $R_a$ of the number of times $a$ occurs within a case, and the distribution $W_a$ of the idle time preceding it
\State Regress the observed case duration $B_c$ on the case-level attributes $\mathbf{x}^{\mathrm{attr}}_c$, and keep the regressor as $\hat{f}_B$ if it beats the median duration $\tilde{B}$ on held-out cases, otherwise set $\hat{f}_B \equiv \tilde{B}$

\State \Return \textbf{(i)} the stochastic Petri net $\mathcal{N} = (N, M_0, M_f, \mathcal{G}, w)$ with the case-behaviour distributions $R_a$ and $W_a$, and \textbf{(ii)} the case-level model $\hat{f}_B$, predicting the total time budget a simulated case is generated to fill
\end{algorithmic}
\end{algorithm}

\begin{algorithm}[H]
\caption{Energy profile extraction.}
\label{algorithm:energy_model_extarction}
\begin{algorithmic}[1]
\Require Table $\mathcal{D}$ with case, activity, timestamps, attributes $\mathbf{x}^{\mathrm{attr}}$, external features $\mathbf{x}^{\mathrm{ef}}$ and energy values; regressors $\mathcal{R}$; maximum number of segments $M_{\max}$

\State \textbf{(A) Extract activity-level energy profiles}
\State For each pair of case $c$ and activity $a$ in $\mathcal{D}$, sort its rows by timestamp and extract their energy values as a raw curve $y_i$ of length $|y_i|$, with the process features $\mathbf{z}_i$ given by $\mathbf{x}^{\mathrm{attr}}$ and $\mathbf{x}^{\mathrm{ef}}$

\State \textbf{(B) Select the reference curve}
\State Resample every raw curve $y_i$ to the median length $S$ of the raw curves of the activity and divide it by its mean value, obtaining its shape
\State Set the reference curve $r$ to the DTW medoid: the raw curve whose shape has the smallest total DTW distance to the shapes of all other curves

\State \textbf{(C) Segment the reference curve}
\State For every number of segments $M \leq M_{\max}$, compute by dynamic programming the best piecewise-constant approximation of the shape of $r$, and keep the $M$ with the lowest BIC, yielding the interior breakpoints $b = (b_1 < \dots < b_{M-1})$ as fractions of the timeline

\State \textbf{(D) Carry the segments onto every curve}
\For{each raw curve $y_i$}
    \State Align the shape of $y_i$ to the shape of $r$ by DTW and map the breakpoints $b$ through the warping path onto the own timeline of $y_i$
    \State Read off its targets: the duration fractions $\phi_{i,1},\dots,\phi_{i,M}$ of the segments, summing to one, and their levels $\lambda_{i,1},\dots,\lambda_{i,M}$, the mean raw energy value within each segment
    \State Add to $\mathcal{E}$ one row with the features $(\mathbf{z}_i,\, a,\, |y_i|)$ and the $2M$ targets, summarising each external-factor series by its mean over the window of the activity
\EndFor
\State Encode the categorical variables and keep the numerical variables unchanged

\State \textbf{(E) Train the segment models}
\For{each segment $m = 1,\dots,M$}
    \State Fit on $\mathcal{E}$ one regressor $h \in \mathcal{R}$ for the duration $\phi_m$ and one for the level $\lambda_m$, under an absolute-error loss
    \State Keep each regressor only if it beats its constant fallback — the training mean of $\phi_m$ and the training median of $\lambda_m$ — on held-out curves
\EndFor
\State Set the curve model $g$ to the resulting bank of $2M$ predictors, and route the activity to a per-position regression instead iff $g$ loses against it clearly on the held-out curves

\State \Return the curve model $g$, the reference curve $r$ and the breakpoints $b$, for each sensor, and activity
\end{algorithmic}
\end{algorithm}

\begin{algorithm}
\caption{Simulation of the process and energy.}
\label{alg:simulation}
\begin{algorithmic}[1]
\Require Stochastic Petri net $\mathcal{N} = (N, M_0, M_f, \mathcal{G}, w)$ with the case-behaviour distributions $R_a$ and $W_a$, and case-duration model $\hat{f}_B$, from Algorithm~\ref{alg:pm_extraction_ml}; curve models $(g, r, b)$ from Algorithm~\ref{algorithm:energy_model_extarction}; production plan $P$ with the attributes $\mathbf{x}^{\mathrm{attr}}_c$ of every planned case and the external factors $\mathbf{x}^{\mathrm{ef}}$; exit discount $\alpha < 1$; step limit $k_{\max}$

\State \textbf{(A) Initialise the case}
\State Take the row of case $c$ from the production plan $P$ and assemble its feature vector $\mathbf{z}^\star$ from $\mathbf{x}^{\mathrm{attr}}_c$ and $\mathbf{x}^{\mathrm{ef}}$
\State Predict the time budget of the case, $B^\star = \hat{f}_B(\mathbf{x}^{\mathrm{attr}}_c)$
\State Set the marking $M \gets M_0$, the elapsed case time $\Delta \gets 0$, the step counter $k \gets 0$ and the fire counters $\nu_a \gets 0$ for every activity $a$
\State Draw the repeat quota of this case, $\kappa_a \sim R_a$, for every activity $a$

\State \textbf{(B) Replay the Petri net}
\While{$M \neq M_f$, $\Delta < B^\star$ and $k < k_{\max}$}
    \State Determine the transitions enabled under $M$ and take their weights from $w$
    \State Multiply by $\alpha$ the weight of the transitions that would lead to the final marking $M_f$, so that the case keeps generating activities while its budget is not spent
    \State Where no budget is used, discount instead the weight of every enabled transition whose activity has already reached its quota, $\nu_a \geq \kappa_a$, as a soft penalty rather than a hard block
    \State Normalise the resulting weights over the enabled transitions, sample $t^\star$ from that distribution and fire it, $M \gets \mathrm{fire}(M, t^\star)$
    \If{$\ell(t^\star) = a$ is a visible activity}
        \State Sample its duration $d^\star$ from $D_{t^\star} \in \mathcal{G}$ and the idle time preceding it, $\delta \sim W_a$
        \State Append the instance $(a,\, \Delta + \delta,\, d^\star)$ to the case log $\mathcal{L}$ and update $\Delta \gets \Delta + \delta + d^\star$ and $\nu_a \gets \nu_a + 1$
    \EndIf
    \State $k \gets k+1$
\EndWhile
\State The case is closed as soon as its budget is spent, at the marking reached at that moment, or earlier if the replay arrives at the final marking $M_f$ on its own

\State \textbf{(C) Predict and place the energy profiles}
\For{each activity instance $(a, \Delta_a, d^\star) \in \mathcal{L}$ and each sensor}
    \State Predict with the model $g$ of that sensor and activity the segment durations and levels of the instance, from the features $\mathbf{z}^\star$ and the duration $d^\star$
    \State Reconstruct $\hat{y}^\star$ on a grid of length $d^\star$: place the segment boundaries at the normalised cumulative durations and fill each segment with the values of $r$, rescaled to its predicted level
    \State Insert $\hat{y}^\star$ into the simulated energy timeline $\hat{Y}$ over the interval $[\Delta_a,\, \Delta_a + d^\star]$
\EndFor

\State \Return the simulated case log $\mathcal{L}$ and the energy timeline $\hat{Y}$, repeating (A)--(C) for every case of the production plan $P$ and superposing the resulting timelines
\end{algorithmic}
\end{algorithm}