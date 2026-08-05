%% 
%% Copyright 2019-2024 Elsevier Ltd
%% 
%% Version 2.4
%% 
%% This file is part of the 'CAS Bundle'.
%% --------------------------------------
%% 
%% It may be distributed under the conditions of the LaTeX Project Public
%% License, either version 1.2 of this license or (at your option) any
%% later version.  The latest version of this license is in
%%    http://www.latex-project.org/lppl.txt
%% and version 1.2 or later is part of all distributions of LaTeX
%% version 1999/12/01 or later.
%% 
%% The list of all files belonging to the 'CAS Bundle' is
%% given in the file `manifest.txt'.
%% 
%% Template article for cas-dc documentclass for 
%% double column output.

%\documentclass[a4paper,fleqn,longmktitle]{cas-dc}
%\documentclass[a4paper,fleqn]{cas-dc}

\documentclass[preprint,10pt,authoryear]{cas-dc}

%\usepackage[authoryear,longnamesfirst]{natbib}
%\usepackage[authoryear]{natbib}
\usepackage[numbers]{natbib}

\usepackage{algorithm}
\usepackage{algpseudocode}

\usepackage{float}

\usepackage{placeins}
\usepackage{caption}

\usepackage{booktabs}
\usepackage{tabularx}
\usepackage{cuted}
\usepackage{array}
\usepackage{caption}


\usepackage{makecell}

\usepackage{dblfloatfix} 




%%%Author definitions
\def\tsc#1{\csdef{#1}{\textsc{\lowercase{#1}}\xspace}}
\tsc{WGM}
\tsc{QE}
\tsc{EP}
\tsc{PMS}
\tsc{BEC}
\tsc{DE}
%%%

\begin{document}
\let\WriteBookmarks\relax
\def\floatpagepagefraction{1}
\def\textpagefraction{.001}
\shorttitle{}
\shortauthors{}

\title [mode = title]{Process and Energy Digital Twin: modeling and simulating industrial processes and their dynamic energy profiles with Process Mining and Machine Learning}                      
\tnotemark[1,2]

%\address[1]{, Street 129, 1043 NX Amsterdam, The Netherlands}


\cortext[cor1]{Corresponding author}
\cortext[cor2]{Principal corresponding author}
\fntext[fn1]{}
\fntext[fn2]{}

\nonumnote{}

\begin{abstract}
Improving the energy efficiency of industrial processes requires models that capture how energy demand is actually generated. Existing data-driven approaches, however, focus on the production process and reduce energy to aggregated values, leaving the continuous energy behavior and its process causes unmodeled. This paper proposes a framework that extracts a Process and Energy Digital Twin of a production facility directly from historical data. Process Mining discovers a granular process model in which time is modeled explicitly, with Machine Learning predicting the duration of every activity and case. The energy behavior is modeled as a continuous profile per activity: its characteristic shape is learned from the historical curves and adjusted to the activity duration, the product, and external factors such as the weather. Coupled, the models translate a production plan into complete, realistic energy profiles. We evaluate the framework on six industrial processes, one synthetic and five from real food-production facilities. The proposed methods outperform stochastic, median-based, and sequence-to-sequence alternatives, and only process-aware simulation reproduces realistic energy profiles. The resulting Digital Twin makes explicit how the process execution generates the energy behavior, deepening the understanding of the industrial system and enabling the simulation of new production plans for process- and energy-aware decision support.
\end{abstract}

\begin{keywords}
Digital Twin \sep Process Mining \sep Machine Learning \sep Energy modeling \sep Energy profiles \sep Discrete-event simulation
\end{keywords}

\maketitle

\section{Introduction}
\label{sec:introduction}

%#1 Domain and importance
Industrial processes account for approximately 40\% of global end-use energy consumption \citep{iea_world_2024}, with emissions primarily stemming from electricity use and industrial process heat. In Europe, for instance, electricity meets about 34\% of industrial energy demand, while the remaining 66\% is dedicated to process heat \citep{de_boer_strengthening_2020}. Driven by international climate targets and net-zero commitments \citep{iea_net_2023}, industries must accelerate the improvement of energy efficiency and the reduction of energy-related emissions.

%#2 Overall problem/situation
Various approaches address these goals, including waste heat recovery \citep{kemp_pinch_2007, klemes_handbook_2022}, electrification of heat generation \citep{ashabi_assessing_2025, knorr_electrification_2025}, electricity peak load management, reduction of power demand, integration of fluctuating renewable energy sources \citep{pee_decarbonization_2018, wenzel_energy-related_2024}, and energy-aware production scheduling \citep{shao_systematic_2024}. However, the effective implementation and optimization of these approaches critically depend on detailed and reliable energy demand models of industrial processes, in order to identify optimization potentials and evaluate their impact. This, in turn, requires a clear understanding of how energy patterns are caused by their underlying process behavior.

%#3 Conclusions of existing literature
A key enabler to address this need is the concept of Digital Twins: software-based representations of physical systems that enable advanced analysis, process improvement, and data exchange between the physical object and the digital copy \citep{liu_digital_2024}. Within Industry 4.0 and 5.0, Digital Twins have become central to modeling and improving industrial processes, with particular emphasis on sustainability and energy efficiency \citep{soori_internet_2023, xu_industry_2021, tao_data-driven_2018}. Given the complexity of production processes, manually modeling the components of such twins is time-consuming and difficult \citep{sargent_verification_2010, nordgren_flexsim_2002, matloff_introduction_2008}. To address this, prior research has increasingly leveraged data-driven approaches—especially Process Mining \citep{van_der_aalst_data_2016}—to automatically extract and simulate process models from historical data, more recently also incorporating energy-related aspects \citep{khodadadi_data-driven_2024, khodadadi_automated_2026}.

%#3 Problem with the current literature
However, these Process Mining-based approaches focus on the process and, where they consider energy at all, still represent it in a simplified manner, e.g., as discrete or aggregated energy consumed during a specific production activity \citep{khodadadi_automated_2026}. Such representations neglect the underlying energy behavior, i.e., the continuous time-series energy profiles and the dynamic effects of the process events on them \citep{wenzel_energy-related_2024}. Consequently, the temporal effects of process execution or external factors like weather on real-time energy behavior remain uncaptured. This limits the ability of Digital Twins to accurately reflect real energy behavior, oversimplifying the production system and constraining their potential to identify and realize meaningful energy efficiency improvements \citep{lal_nathan_s_accounting_2018, gonzalez_process_2025}.

%#5 How this study addresses the gap
To address this gap, we propose a data-driven process and energy modeling framework for industrial systems that integrates Process Mining methods \citep{van_der_aalst_data_2016} with causally grounded, continuous energy profiles trained with Machine Learning (ML) methods \citep{james_introduction_2021}. Together, these components form the foundation for a facility's Process and Energy Digital Twin. By modeling discrete process execution with continuous energy behavior from historical data, the framework captures complex real-world dynamics—including interactions among process events, product attributes, external factors, and energy behavior. Ultimately, the Digital Twin enables a more realistic modeling and understanding of the manufacturing system and its accurate simulation.

%#6 Study setting, data, and methods
Our methodology extracts the process model with Process Mining algorithms, preserving the properties that cause the energy behavior: the activities are modeled granularly, down to the machine activities that generate the energy profiles, and time is modeled explicitly, with Machine Learning (ML) algorithms predicting the duration of each activity and each production case. On top of this, we extract a continuous energy profile model per activity and sensor: the historical curves are aligned with Dynamic Time Warping (DTW), and the durations and levels of their characteristic phases are learned with ML algorithms from product attributes and external factors such as the weather. Coupled, these models form the Process and Energy Digital Twin, supporting the understanding of the process and energy behavior, its simulation, and what-if analysis of new production plans.

We validate the framework in three evaluations—of the process, the individual energy profiles, and the complete profiles of simulated cases—on six industrial processes: one synthetic with known ground truth and five from two real food-production facilities, comprising 766 cases and 79 energy sensors over close to 7{,}000 hours of production.

The results show that the extracted Process and Energy Digital Twin reproduces both the structure and the temporal behavior of the process and its energy profiles. The process is modeled accurately when, beyond its control flow, time is modeled explicitly—the durations of the individual activities and the total duration of each case, which bounds the simulation—and this temporal fidelity is what the energy modeling builds on. The energy profiles, in turn, are predicted most realistically from the characteristic shape of each activity's curve, adjusted to the process execution and the external factors, outperforming the evaluated alternatives. The best complete profiles come from coupling the two models: approaches that skip the process match aggregate metrics but produce shapes detached from the real execution. Grounding the energy behavior in the process that causes it is thus what makes the simulated profiles realistic.

%contribution
With this, the paper contributes to the Process Mining literature \citep{van_der_aalst_data_2016, van_der_aalst_process_2022}, especially the work on data-driven simulation and Digital Twins in manufacturing \citep{lugaresi_automated_2023, castiglione_automated_2024}, showing that activity granularity and explicit time modeling are what make process models usable for posterior energy modeling. To ML-based energy modeling \citep{he_generic_2020, worrlein_using_2024, gonzalez_process_2025, khodadadi_automated_2026}, we contribute a method that learns continuous, activity-level energy profiles conditioned on product attributes and external factors. For practitioners, the framework is a blueprint for building realistic twins from data they already record, supporting process and energy-aware decision-making.

% #9 Paper outline
The remainder of this paper is structured as follows: Section~\ref{sec:related-work} introduces important background concepts and reviews the related work. Section~\ref{sec:methodology} presents the proposed Process and Energy Digital Twin framework, describing its extraction from real-world data and its application for simulation. Section~\ref{sec:evaluation} describes the evaluation procedure. Section~\ref{sec:results} reports the case study and evaluation results, followed by the discussion (Section~\ref{Discussion}) and conclusions, limitations and future work (Section~\ref{Conclusion, Limitations, and Future Work}).

\section{Background and Related Work}
\label{sec:related-work}
%\subsection{Digital Twin and simulation}

To investigate a system such as a manufacturing plant, its structural and behavioral assumptions must first be formalized into a \textit{model}—for instance as mathematical formulas or logical relationships \citep{wainer_discrete-event_2017}. When such a model is software-based, it is referred to as a \textit{digital model} \citep{wainer_discrete-event_2017}. Through simulation, a computer dynamically executes such a digital model to imitate the system's real-world operation over time, generating data that can be used to analyze and optimize its performance \citep{law_simulation_2015}. Beyond driving an isolated simulation, a digital model can also be coupled with the physical system it represents through an exchange of real-world data. If an automated, bidirectional data flow is established between the physical system and the digital model, it is considered a \textit{Digital Twin} \citep{grieves_digital_2014, liu_digital_2024}.

Several simulation paradigms exist to implement the digital model underlying a Digital Twin, each suited to different system characteristics. Dynamic simulation addresses systems that evolve continuously over time and is common in physical and engineering contexts \citep{cellier_continuous_2013}, while discrete-event simulation (DES) is widely used for systems whose state changes at distinct points in time \citep{wainer_discrete-event_2017}. Further paradigms include agent-based simulation, which models the interactions of autonomous agents \citep{bonabeau_agent-based_2002}, system dynamics, a continuous approach emphasizing feedback loops and accumulations \citep{bala_system_2017}, and hybrid combinations of these.

As manufacturing systems typically exhibit both discrete process dynamics and continuous energy behavior, this paper focuses on discrete-event and dynamic simulation to capture these two aspects within a Digital Twin. To this end, we integrate data-driven methods with established discrete and continuous modeling techniques. The subsequent sections detail these two paradigms and explore their respective data-driven modeling approaches.

\subsection{Discrete Event Simulation and Process Mining}

Discrete event simulation (DES) is a method used to capture the behavior of a system by advancing its state only at discrete points in time, driven by events such as arrivals, service completions, or failures. Instead of evolving continuously, the system transitions between states when events occur, and these events are typically managed through an event-scheduling mechanism (e.g., a future event list) \citep{wainer_discrete-event_2018}. DES models usually consist of entities (e.g., jobs or customers), resources (e.g., machines or servers), queues, and routing logic that governs how entities move through the system. This paradigm is widely used to represent systems such as manufacturing processes, logistics networks, and service operations \citep{wainer_discrete-event_2017}.

For a DES, it is necessary to create a digital model of the system by defining key components, such as entities, resources, and queues—alongside the logic governing their interactions. Modelers must specify potential events and implement a scheduling mechanism to determine the sequence and timing of state changes \citep{wainer_discrete-event_2018,brailsford_discrete-event_2014}. Although these models are typically built using dedicated software \citep{nordgren_flexsim_2002, matloff_introduction_2008}, the development process remains challenging and time-consuming. This difficulty arises from the need to explicitly specify stochastic behavior, routing rules, and resource constraints, which often results in complex models \citep{sargent_verification_2010}. As the system's complexity grows, models become harder to build, validate, and maintain. These issues can limit scalability and hinder the practical adoption of DES in complex real-world systems.

To reduce this effort and better reflect real-world production, data-driven approaches, particularly \textit{Process Mining (PM)}, have gained increasing attention. PM is a data-driven approach that uncovers and analyzes the actual execution of processes based on historical real production data \citep{daniel_process_2012, van_der_aalst_process_2022}. In particular, \textit{Process Discovery}, one of the core techniques within PM, can automatically derive process models from such data. Process discovery approaches capture both the model structure and its parameters, including control-flow patterns, resource behavior, and performance characteristics \citep{dumas_fundamentals_2018,weske_business_2019, dreher_application_2021, van_der_aalst_process_2022, rozinat_discovering_2009, camargo_automated_2020, castiglione_automated_2024}. In smart manufacturing production systems, this level of automation is commonly regarded as a prerequisite for complex operational Digital Twins capable of keeping pace with frequent changes in production routings and plans \citep{friederich_framework_2022,uhlemann_digital_2017,zheng_application_2019}.

The data source for PM is an \textit{event log} $L$: a set of recorded events capturing the step-by-step execution of the system. Every event $e$ from the universe of events $E$ carries a timestamp $\pi_{time}(e)$ and refers to a case $c \in C$ (a process instance, e.g., the complete flow of a product through the manufacturing line) via $\pi_{case}(e) = c$ and to an activity $a \in A$ via $\pi_{act}(e) = a$. The events of a case, ordered by time, form a trace $\sigma = \langle e_1, \dots, e_n \rangle$ with $\pi_{time}(e_i) \leq \pi_{time}(e_{i+1})$. Events can carry further attributes, such as the executing resource $\pi_{res}(e)$, a lifecycle transition $\pi_{life}(e)$ (e.g., \textit{start} or \textit{complete}), or domain-specific data $\pi_{data}(e)$, e.g., the size of an order \citep{daniel_process_2012, van_der_aalst_process_2016, van_der_aalst_process_2022}.

In this regard, process models can be extracted using process discovery algorithms. The Alpha Miner \citep{van_der_aalst_workflow_2004}, one of the earliest approaches, identifies basic sequential, parallel, and causal relations between activities but is sensitive to noise. The Heuristic Miner \citep{weijters_process_2006, weijters_flexible_2011} improves robustness through frequency-based relations and is thus better suited to noisy event data. The Inductive Miner \citep{leemans_discovering_2013, leemans_scalable_2015} is among the most widely used algorithms, as it generates structured and sound process models that are relatively easy to interpret.

Depending on the method and implementation, process discovery algorithms can produce various process model representations, including heuristic nets, process trees, directed-follow graphs, Business Process Model and Notation, and Petri nets \citep{van_der_aalst_data_2016}. Among these, Petri nets are the most common representation for models that can be directly utilized for simulation in manufacturing systems \citep{castiglione_automated_2024, bause_stochastic_2002, simon_adapting_2018}.

Petri-net-based models capture the core components of a process: \textit{transitions} correspond to the observed activities of the event log, \textit{places} represent the conditions that enable and synchronize them, and together they encode the control flow — causal dependencies, sequential execution, and parallelism — as well as the activity durations \citep{bause_stochastic_2002}. Extensions address specific system complexities: Stochastic Petri nets (SPNs) model the control-flow variability of manufacturing processes through probabilistic transitions and exponentially distributed firing delays \citep{bause_stochastic_2002, simon_adapting_2018, khodadadi_automated_2026}, and Stochastic Timed Petri nets (STPNs) \citep{wang_timed_2012} generalize this by allowing arbitrary firing-delay distributions, so that activity durations can follow empirically fitted distributions. In this work, we use STPNs as basis:

A STPN is a tuple $\mathcal{N} = (N, M_0, M_f, \mathcal{G}, \pi)$, where $N = (P, T, F, W)$ is the underlying Petri net: $P$ and $T$ are disjoint finite sets of places and transitions, $F \subseteq (P \times T) \cup (T \times P)$ is the flow relation, and $W$ assigns arc weights, with $W(x,y) > 0$ if and only if $(x,y) \in F$. The functions $M_0, M_f : P \rightarrow \mathbb{N}$ denote the initial and final markings. A transition $t \in T$ is \emph{enabled} at marking $M$ if $M(p) \geq W(p,t)$ for all $p$ in its preset $\bullet t$; when it fires, the marking updates to $M'(p) = M(p) - W(p,t) + W(t,p)$ for all $p \in P$, and the execution terminates once $M_f$ is reached.

The \textit{stochastic} part is the routing function $\pi : T \rightarrow [0,1]$: among mutually conflicting enabled transitions $\mathcal{C}$ (XOR-splits), the firing transition is sampled from $\pi$, with $\sum_{t \in \mathcal{C}} \pi(t) = 1$. In this work, $\pi$ is induced by non-negative transition weights $w$, $\pi(t) = w(t) / \sum_{t' \in \mathcal{C}} w(t')$. The \textit{timed} part is $\mathcal{G} = \{D_t\}_{t \in T}$: when $t$ fires, its duration is sampled from $D_t$, an arbitrary distribution fitted to observed data.


\subsection{Dynamic Simulations and data-driven Energy Modeling}

Dynamic simulation executes system models whose state evolves over time. In contrast to DES, where the state changes only at discrete event occurrences, a dynamic model describes how state variables, quantities such as temperatures, flows, or power, change continuously in response to inputs and the system's internal dynamics, producing trajectories of the system behavior over the timeline \citep{cellier_continuous_2013, law_simulation_2015}. Within this broad category, mathematical models can be classified according to how time is represented: continuous-time models and Discrete-time models.

In continuous-time models, the state variables evolve smoothly and are governed by differential equations \citep{cellier_continuous_2013}. By treating system dynamics as an uninterrupted temporal process, such models are well suited to domains in which variables change without discontinuity, including mechanical, chemical, biological, and energy systems. Typical applications range from fluid dynamics and thermal processes to mechanical motion. Continuous modeling is likewise established in energy systems, where quantities such as power, temperature, and pressure vary continuously over time \citep{grigsby_power_2007, bergman_fundamentals_2011}.

Discrete-time models, by contrast, represent time as a sequence of steps and describe the system evolution through difference equations. They are widely used in engineering applications, particularly in digital control, where computations are carried out at fixed sampling intervals and require finite time to determine the next state \citep{brunton_data-driven_2022}. Such models may be inherently discrete, or they may be obtained by discretising a continuous-time formulation \citep{cellier_continuous_2013}.

Constructing dynamic models of energy systems requires deriving governing equations from first principles, such as conservation laws (mass, energy, momentum) and thermodynamic relations \citep{brunton_data-driven_2022, bergman_fundamentals_2011}. In practice, this approach demands deep domain expertise and a thorough understanding of the underlying physical mechanisms. Although, these models are typically developed using dedicated dynamic simulation software \citep{klee_simulation_2018}, this task can be labor-intensive. Furthermore, it introduces modeling errors through unavoidable simplifying assumptions, particularly when representing complex systems with poorly understood physical dynamics \citep{ghadami_data-driven_2022}.

These limitations have steered interest into data-driven or \textit{Machine Learning (ML)} modeling techniques, which extract a system's model directly from historical measured data in the form of discrete-time models \citep{brunton_data-driven_2022}. For instance, regression and time-series algorithms can effectively approximate complex, nonlinear energy system dynamics without explicit knowledge of the underlying physical equations \citep{bishnu_computational_2023, van_den_hof_system_2020}. 

The use of ML techniques has been explored to predict the total energy consumption of production facilities \citep{mosavi_energy_2019}, as well as energy consumption at the machine level \citep{he_generic_2020, zhang_data-driven_2021, mawson_deep_2020}. In general, these approaches can be described as supervised learning problems in which a target variable \(y\) is modeled as a function of a set of input variables \(\mathbf{x}\). Depending on whether the model predicts a single value or a time-varying output, this takes two forms:
\begin{equation}
\label{eq:ml_energy}
y = f(\mathbf{x}) + \varepsilon, \qquad \text{or} \qquad y_t = f(\mathbf{x}_t) + \varepsilon_t,
\end{equation}
where in the first case $y$ is a scalar target predicted from a static input vector $\mathbf{x}$, and in the second case $y_t$ and $\mathbf{x}_t$ denote the output and input at the same time step $t$, respectively. 


%A note on terminology: running a dynamic model forward in time is conventionally called "simulation" in the previous cited studies, whereas obtaining the same kind of output from an ML model is conventionally called "prediction", even when the ML model is used for a simulative purpose, such as playing out a scenario of input parameters and observing the result \cite{von_rueden_combining_2020}. We follow the convention and refer to the outputs of the ML energy-profile models as "predictions", although in this work they play the same role as a dynamic simulation.
 
\subsection{Related Work}

Our work bridges the fields of discrete-event and dynamic simulation, specifically focusing on data-driven modeling techniques such as PM and ML. To contextualize our contribution, Table~\ref{tab:literature_comparison} summarizes related literature across five dimensions: the use of process simulation, the application of Process Mining, general energy considerations (for instance as a total value at the end of a simulation), the ability to simulate energy profiles, and the use of data-driven energy profile modeling.

Studies have worked on automated process model extraction, generating process models from historical event logs with PM. For instance, \cite{friederich_framework_2022} applied PM techniques to model material flows and machine reliability, demonstrating that these models enable data-driven Digital Twins for manufacturing system simulations. Similarly, \cite{lugaresi_automated_2023} proposed a PM algorithm to construct Digital Twins for systems with complex material flows, validating its effectiveness in a real-world manufacturing environment. Further expanding on this, \cite{castiglione_automated_2024} introduced an event-centric process mining framework that tracks material entering and leaving machines, enabling the rapid, automated generation of digital models while remaining robust under low-data conditions.

Other studies have used DES to include energy consumption data. For example, an early approach is the EnergyBlocks methodology \citep{weinert_methodology_2011}, which composes measured power-demand segments per machine operating state according to the production plan; as the segments are fixed recordings, the profiles do not adapt to product characteristics or external conditions. Also, \cite{kohl_discrete_2014} expanded DES with energy models to generate full energy profiles rather than fixed values, showing that this improves energy predictions for production lines and full factories. Similarly, \cite{kouki_input_2017} reviewed the literature on incorporating load profiles into discrete event simulations and proposed using stochastic distributions. They showed that their approach results in only a small deviation from actual energy measurements.

Furthermore, a few studies have combined discrete event data or simulations with ML to predict continuous energy profiles. For instance, \cite{woerrlein_method_2020} applied sequence-to-sequence ML models to predict time-series energy consumption directly from numerical control (NC) code. Building on this, \cite{worrlein_using_2024} showed that sequence-to-sequence models can capture energy curve patterns effectively, which significantly improves prediction accuracy.

Finally, other studies use PM to extract process models directly from event data and add specific energy-related factors, such as power consumption, waste generation, and CO2 emissions. For instance, \cite{khodadadi_data-driven_2024} developed energy-oriented Digital Twins to better understand energy behavior in smart factories, showing how operational schedules interact with energy use. Building on this, \cite{khodadadi_automated_2026} proposed a framework to automatically extract Petri nets with multidimensional properties like time, energy consumption, and waste generation. In their experiment, they showed that their approach can simulate what-if scenarios and reduce energy consumption without affecting production output. Similarly, \cite{gonzalez_process_2025} extracted process models from manufacturing event logs and combined them with energy data for better heat recovery potential analysis.


\begin{table*}[H]
\centering
\caption{Comparison of the proposed approach with related literature}
\label{tab:literature_comparison}
\resizebox{\textwidth}{!}{%
\begin{tabular}{@{}p{4.5cm}p{7cm}ccccc@{}}
\toprule
\textbf{Study} & \textbf{Description} & \textbf{Process simulation} & \textbf{Process Mining} & \textbf{Energy consideration} & \textbf{Energy profiles simulation} & \textbf{Data-driven energy profile modeling} \\ \midrule

\cite{weinert_methodology_2011} & Energy profiles composed of fixed measured segments per machine operating state. & \checkmark & -- & \checkmark & \checkmark & -- \\ \addlinespace
\cite{kohl_discrete_2014} & DES extending material flows with energy consumption information. & \checkmark & -- & \checkmark & \checkmark & -- \\ \addlinespace
\cite{kouki_input_2017} & DES modeling energy via stochastic distributions. & \checkmark & -- & \checkmark & \checkmark & -- \\ \addlinespace
\cite{woerrlein_method_2020} & DES triggers a predictive model curve. & \checkmark & -- & \checkmark & \checkmark & \checkmark \\ \addlinespace
\cite{friederich_framework_2022} & Automatic process Digital Twin generation with PM. & -- & \checkmark & -- & -- & -- \\ \addlinespace
\cite{camargo_learning_2023} & Automated generative DES discovery from event logs with PM. & \checkmark & \checkmark & -- & -- & -- \\ \addlinespace
\cite{lugaresi_automated_2023} & Automatic discovers simulation models from event logs. & \checkmark & \checkmark & -- & -- & -- \\ \addlinespace
\cite{castiglione_automated_2024} & Event-centric PM for generating automated Digital Twins. & -- & \checkmark & -- & -- & -- \\ \addlinespace
\cite{belina_ethospenalps_2024} & Open-source tool for load profile simulation. & \checkmark & -- & \checkmark & \checkmark & -- \\ \addlinespace
\cite{worrlein_using_2024} & Predict energy profiles from a discrete simulation. & \checkmark & -- & \checkmark & \checkmark & \checkmark \\ \addlinespace
\cite{khodadadi_data-driven_2024} & PM extracting stochastic nets that also calculates total energy consumption. & \checkmark & \checkmark & \checkmark & -- & -- \\ \addlinespace
\cite{gonzalez_process_2025} & PM for creating process model and synthetic energy profiles from punctual energy data. & -- & \checkmark & \checkmark & -- & -- \\ \addlinespace
\cite{khodadadi_automated_2026} & Multi-flow PM for total energy and waste scalar predictions. & \checkmark & \checkmark & \checkmark & -- & -- \\ \midrule
\textbf{Our approach} & \textbf{Extracts process models with Process Mining and energy models with Machine Learning. These models form a Process and Energy Digital Twin that can be used for understanding and simulation.} & \textbf{\checkmark} & \textbf{\checkmark} & \textbf{\checkmark} & \textbf{\checkmark} & \textbf{\checkmark} \\ \bottomrule
\end{tabular}%
}
\end{table*}

As summarized in Table~\ref{tab:literature_comparison}, the literature covers some individual ingredients, but not their combination: data-driven process modeling without an energy perspective \citep{camargo_discovering_2021, lugaresi_automated_2023, castiglione_automated_2024}, energy profile prediction that is not coupled to a data-driven process model \citep{weinert_methodology_2011, kohl_discrete_2014, kouki_input_2017, woerrlein_method_2020, worrlein_using_2024}, and process-aware approaches that reduce energy to aggregate indicators or punctual values \citep{khodadadi_data-driven_2024, khodadadi_automated_2026, gonzalez_process_2025}. However, process execution determines when and how energy is consumed, and the energy dynamics in turn reflect the underlying process behavior, so neither can be modeled meaningfully in isolation. This paper addresses this gap: we introduce a framework that jointly extracts process and energy models from historical data, process and energy data, production plans, and external factors, and couples them into a Process and Energy Digital Twin for manufacturing. To the best of our knowledge, this is the first approach that models both the process and its energy profiles in a purely data-driven way. The following section details the proposed methodology.

\section{Methodology}
\label{sec:methodology}
\subsection{Framework}

Figure~\ref{fig:framework} presents the proposed framework for automatically deriving integrated process and energy models from real-world production data. Within this framework, industrial operations and energy profiles are conceptualized through causal dependencies: process execution is driven by the production plan and external factors, while the resulting energy behavior is driven by process events and, likewise, external factors. External factors comprise exogenous variables not directly captured by event logs or energy sensor measurements, such as weather conditions, seasonality (e.g., month or day of the week), personnel availability, energy tariffs, and machine wear. The framework is structured into four phases:

\begin{figure*}[H]
    \centering
    \includegraphics[width=1\textwidth]{1_framework.pdf}
    \caption{Overview of the proposed Process and Energy Digital Twin building.}
    \label{fig:framework}
\end{figure*}
%\vspace{-25mm}

First, in \textit{Data Collection}, the production plan, external factors, process data, and energy data are gathered from the respective IT systems of the manufacturing facility. Second, the production plan, process data, and external factors are passed to \textit{Process Model Extraction}, where PM is applied to derive the process model. Third, process data, energy data, and external factors are simultaneously provided to the \textit{Energy Model Extraction} phase, where ML models predict segment durations and levels, while stored reference curves define the shape of the corresponding energy profiles. Fourth, the resulting process and energy models jointly constitute the Digital Models of Process and Energy, which form the core of the \textit{Analysis} phase. In this final phase, the models can be inspected directly or serve as the basis for simulation, enabling descriptive, diagnostic, predictive, and prescriptive analytics that support or directly inform decisions to improve real-world production. The following subsections describe the individual components of the framework in more detail.

\subsection{Data Collection}

In manufacturing, process data originates from sources such as Enterprise Resource Planning (ERP), Manufacturing Execution Systems (MES), and Supervisory Control and Data Acquisition (SCADA) systems, as well as Programmable Logic Controllers (PLCs) at the machine level. These sources capture both the material flow, such as products entering and leaving a machine, and the machine states involved in transforming a product. As the data comes from multiple systems, it must be aligned into a coherent event log—synchronizing time formats, removing duplicates, and ensuring chronological ordering—before it can be used for process discovery.

Energy data is typically obtained from Energy Management Systems, SCADA systems, or PLCs. We deliberately refer to \textit{energy behavior} rather than energy demand alone: the sensors capture not only direct demands, such as electric power or steam flow, but also related quantities, such as temperatures or pressures, which characterize the energetic state of the process and can be modeled analogously. Both event and energy data are usually recorded continuously for traceability, energy management, and process control. Preprocessing includes verifying measurement units, correcting implausible values, and—since energy data is often recorded at high resolution (e.g., milliseconds) for control purposes—aggregating it to the level of detail needed for modeling.

Data on external factors can be obtained from company IT systems or weather services. As their impact varies across processes—heating ambient air, for instance, demands more energy in winter—the relevant factors must be identified per process. They also often differ between energy and process flow: weather typically affects energy but not the process.

\subsection{Process Model Extraction}

Figure \ref{fig:process_model_extraction} shows the workflow for process model extraction, based on three data sources: process data, production plan, and external factors. As shown in the upper part of the figure, the data is visualized and prepared at two levels—the machine level and the material-flow level—and then integrated into a unified table, the prepared process data, which serves as the input for PM to extract the \textit{process model}. This integration is necessary for two reasons: the process must be modeled at a \emph{granular} level, so that each activity has a characteristic and therefore learnable energy curve, and \emph{time} must be modeled explicitly, as the duration of the individual activities and the total duration of a case, since both determine when and how the energy curves unfold. These two requirements drive the data integration described below and the design of the extraction algorithm at the end of this section.

\begin{figure*}[H]
    \centering
    \includegraphics[width=1\textwidth]{2_process_model_extraction.pdf}
    \caption{Process model extraction.}
    \label{fig:process_model_extraction}
\end{figure*}

Most manufacturing process modeling approaches focus only on material flow \citep{camargo_discovering_2021, kouki_input_2017, rozinat_discovering_2009, castiglione_automated_2024}, while others model material flow and machine states separately \citep{friederich_data-driven_2022, friederich_framework_2022, friederich_process_2022, gonzalez_process_2025}. Neither is sufficient to achieve the granularity required for energy-profile modeling: for this, both perspectives must be linked. The production plan initiates the material flow, determining the routing of products through the production system and when machines are triggered to process them, with the corresponding events recorded in a \textit{material-flow event log}. The triggering of a machine and the characteristics of the product being processed, in turn, affect the machine's activities, internal operational modes, and equipment states during processing, which are recorded in a \textit{machine event log}. It is these fine-grained machine activities that ultimately shape the resulting energy profiles.

The two types of event logs can be obtained from different IT systems or jointly from a single one, but it is important to keep them logically separate. Since visualizing both perspectives together can produce complex ``spaghetti-like'' models \citep{castiglione_automated_2024}, we propose using them separately for \textit{visualization} but integrating them for \textit{modeling}.

We illustrate this using a hospital infusion-bag production line, in which bags are filled, sterilized, and packaged (see Figure \ref{fig:process_model_extraction}). A \textit{material-flow event log} tracks the movement of batches as they enter and leave the filling, sterilization, and packaging machines, while a \textit{machine event log} records the sterilization operations decomposed into preparation, heating, and cooling activities, each with a characteristic energy profile. For \textit{visualization}, process discovery is executed independently for the material flow and for each machine's event log, yielding, for instance, heuristic nets that represent the execution of the process, depicted below their respective tables.

For the \textit{modeling} of the process and its energy behavior, the level of detail must match the granularity of the underlying energy profiles: this is the first of our two arguments. Each profile is determined by the execution of a machine activity on a given product and external factors at that moment. Modeling at this granular level yields more accurate results than modeling at the material-flow level, which can span several machine activities and therefore lacks a single characteristic curve to learn.

Figure~\ref{fig:machine_activities} illustrates this point using the sterilization step of the infusion-bag example. It shows two runs of the same material-flow activity, each composed of distinct machine activities (\textit{prepare}, \textit{heat}, \textit{cool}) with their own characteristic energy signature that recurs whenever the activity is executed — the granularity at which the energy profile is actually generated. In the first run, both executions follow the full sequence and yield nearly identical profiles, whereas in the second run the second execution omits the preparation step and starts directly with \textit{heat}. Despite the identical material-flow activity label, the profiles differ substantially in shape and magnitude, and the second run is noticeably shorter in total duration. Modeling solely at the material-flow activity level (or at the level of the whole production case) would therefore not allow accurate prediction of the shape.

The second argument concerns \textit{time}. In existing process modeling approaches that consider energy, durations are typically obtained by fitting a statistical distribution to the observed activity durations \citep{khodadadi_automated_2026}. If the goal is to predict a complete energy curve, however, the duration of an activity directly shapes its energy profile. We therefore predict durations with ML algorithms from inputs such as activity type, process attributes, and external factors, capturing the underlying variability better than a pure distribution fit. Time matters equally at the case level: the total duration of a case determines how its activities, and with them their energy profiles, are placed over the timeline, so we also predict it and use it to bound the total time of the simulation.

\begin{center}
    \includegraphics[width=1\columnwidth]{4_machine_activites.pdf}
    \captionof{figure}{Material flow and machine activities. The profile shows different characteristic behaviors across the machine states.}
    \label{fig:machine_activities}
\end{center}

This activity-level granularity also generalizes across process types. PM is most commonly applied in discrete manufacturing, where each product unit–an infusion bag, a household appliance, a car–carries its own case\_id and can be traced through the process \citep{friederich_data-driven_2022, camargo_discovering_2021}. Many energy-intensive industries, however, run continuous processes, as in chemical or food production, designed for high resource utilization and stable flows \citep{kemp_pinch_2007}. There, a production order can serve as the case\_id, grouping material that shares a recipe \citep{gonzalez_process_2025}; the case duration is then largely bound by the amount produced, and the material flow often reduces to a single activity. Even where the material flow is continuous, the machine activities remain discrete–they have start and end times and can be modeled as process activities.

\paragraph{Process model extraction algorithm.} The two arguments above translate directly into the extraction and training pipeline summarized in Algorithm~\ref{alg:pm_extraction_ml}: it splits activities down to the granularity at which each has a characteristic energy curve, and it models time explicitly, predicting the duration of every activity instance as well as the total duration of a case, which later bounds the simulation with a budget.

\begin{algorithm}
\caption{Extraction of the process model.}
\label{alg:pm_extraction_ml}
\begin{algorithmic}[1]
\Require Event log $L$ over cases $C$ and activities $A$, each event carrying its object, attributes $\mathbf{x}^{\mathrm{attr}}$ and external features $\mathbf{x}^{\mathrm{ef}}$ via $\pi_{data}(e)$; process miners $\mathcal{M}$ with configuration spaces $\Theta_m$; parametric duration families $\mathcal{P}$; ML algorithms $\mathcal{R}$; minimum sample size $n_{\min}$

\State \textbf{(A) Prepared process data.}
\State Split every activity with nested sub-activities until each one has a characteristic energy profile, and assemble $PT$ with one row per event: $(\pi_{case}(e),\; \pi_{act}(e),\; \pi_{time}(e),\; \mathbf{x}^{\mathrm{attr}},\; \mathbf{x}^{\mathrm{ef}})$

\State \textbf{(B) Petri net structure} $(N, M_0, M_f)$
\State For each sub-log induced by an object and its higher-level activity, and for every miner $m \in \mathcal{M}$ and configuration $\theta \in \Theta_m$, mine the Petri net $(N_{m,\theta},\, M_{0,m,\theta},\, M_{f,m,\theta})$ with labeling $\ell : T \rightarrow A \cup \{\tau\}$, score $q(m,\theta)$ as the mean of replay fitness, precision, generalization and simplicity, and set $(N, M_0, M_f, \ell)$ to the Petri net maximizing $q$

\State \textbf{(C) Firing-time distributions} $\mathcal{G}$
\For{each activity $a \in A$ with $n_a$ observed durations $d$}
    \State Fit every family in $\mathcal{P}$ to $d$ by maximum likelihood estimation and keep as $\hat{p}_a$ the one with the highest Kolmogorov--Smirnov $p$-value
    \If{$n_a \geq n_{\min}$}
        \State Fit each $h \in \mathcal{R}$ on $d$ by $K$-fold cross-validation, with features from $\mathbf{x}^{\mathrm{attr}}$, $\mathbf{x}^{\mathrm{ef}}$, the position of $a$ in the trace, the elapsed case time and the lagged durations; keep $\hat{f}_a = \arg\min_{h} \mathrm{MAE}_{\mathrm{CV}}(h)$ and rescale it by its out-of-fold ratio $\bar{d} / \bar{\hat{d}}^{\mathrm{oof}}$, so that it predicts the mean of the right-skewed durations
    \EndIf
\EndFor
\State Fit the global model $\hat{f}_{\mathrm{global}}$ likewise on all activities pooled, with the median duration of the activity as an additional feature, and set $\mathcal{G} = \{D_t\}_{t \in T}$, with $D_t$ immediate for every $\tau$-transition and, for every visible transition, either sampled from $\hat{p}_{\ell(t)}$ or predicted by $\hat{f}_{\ell(t)}$ or $\hat{f}_{\mathrm{global}}$

\State \textbf{(D) Routing weights} $w$\textbf{, case behavior and duration budget}
\State Set the routing weight $w(t)$ of every transition to the number of times it fires in the token replay of the training log, and $w(t) = 1$ for transitions that are never replayed, so that unobserved behavior keeps a small residual probability
\State For every activity $a$, record the empirical distribution $R_a$ of the number of times $a$ occurs within a case, and the distribution $W_a$ of the idle time preceding it
\State Regress the observed case duration $B_c$ on the case-level attributes $\mathbf{x}^{\mathrm{attr}}_c$, and keep the ML model as $\hat{f}_B$ if it beats the median duration $\tilde{B}$ on held-out cases, otherwise set $\hat{f}_B \equiv \tilde{B}$

\State \Return \textbf{(i)} the stochastic Petri net $\mathcal{N} = (N, M_0, M_f, \mathcal{G}, w)$ with the case-behavior distributions $R_a$ and $W_a$, and \textbf{(ii)} the case-level model $\hat{f}_B$, predicting the total time budget for the simulation
\end{algorithmic}
\end{algorithm}


\textbf{Step A: Prepared process data.} The material-flow and machine-activity event logs are merged and flattened at the lowest available granularity; where no finer machine-level granularity exists for an activity, the material-flow activity itself is used instead. Each activity instance is then enriched with case attributes and external factors, such as time of day or ambient conditions. This is the prepared process data necessary for modeling.

\textbf{Step B: Petri net structure.} Process discovery is applied independently of the chosen mining algorithm: rather than fixing one miner and one parameter setting, every miner is run over a grid of possible configurations—for instance, the noise threshold of the Inductive Miner, or the dependency and thresholds of the Heuristic Miner—and each discovered Petri net is scored on four standard quality criteria: fitness, precision, generalizability, and simplicity \citep{buijs_quality_2014}. The four criteria are averaged into one score, and the best-scoring configuration across all miners is retained.

\begin{figure*}[H]
    \centering
    \includegraphics[width=1\textwidth]{3_energy_models_extraction.pdf}
    \caption{Energy model extraction.}
    \label{fig:energy_model_extraction}
\end{figure*}

\textbf{Step C: Firing-time distributions.} A duration is assigned to every activity using ML models trained on case attributes, external factors, and process-level features such as the position of the activity in the case, the elapsed case time, and the recent lagged durations. Two strategies are compared: a \textit{local} model per activity and a \textit{global} model over all activities that uses the activity identity as a feature. In both, the model with the lowest cross-validated mean absolute error is retained. Since durations are right-skewed, predictions are rescaled by the ratio between observed and out-of-fold mean duration to correct for the median-tracking bias of the underlying models. Activities with too few instances fall back to the median duration of the activity.

\textbf{Step D: Routing weights, case behavior, and duration budget.} The stochastic Petri net is completed with routing weights derived from token replay, and a case-level time budget is added to bound the simulation. The budget is predicted by an ML model from case attributes and is kept only if it outperforms the median case duration on held-out cases. In addition, two distributions are recorded for every activity: how often it occurs within a case, keeping the repetition count of simulated cases in a realistic range instead of leaving loops to an unbounded random choice, and how long the process idles before it starts, reproducing the waiting time between consecutive activities.

The algorithm returns two components: a stochastic Petri net, which describes the control flow, the probability of each routing decision, and the repetition and waiting behavior of the cases, with duration models are attached to its transitions; and a case-duration model, which gives the time budget a simulated case is generated to fill. 

Both can be fed with a production plan and external factors to generate process data that reflect the control flow, operational constraints, and temporal dynamics of the real process–granular in its activities and explicit in its durations–and therefore suitable as a basis for the energy modeling that follows.

\subsection{Energy Model Extraction}

Figure~\ref{fig:energy_model_extraction} shows the workflow for the energy model extraction, from the raw sensor measurements on the left to the trained energy models on the right. The task of these models is to predict a dynamic energy profile from discrete process information, such as the activity, its duration, product characteristics, and external factors. In contrast to the formulations of Equation~\ref{eq:ml_energy}, which predict a scalar total or a value per time step from time-varying inputs, here the model must generate the complete curve directly from a static input descriptor $\mathbf{x}$:
\begin{equation}
\hat{\mathbf{y}} = (\hat{y}_{1}, \dots, \hat{y}_{T})^{\top} = f(\mathbf{x}) + \boldsymbol{\varepsilon},
\end{equation}
where $\hat{\mathbf{y}} \in \mathbb{R}^{T}$ is the predicted energy profile over $T$ time steps and $\boldsymbol{\varepsilon} \in \mathbb{R}^{T}$ is the error vector. This formulation is substantially more challenging than the previous ones, as a small set of discrete inputs can plausibly account for only a limited number of characteristics of a curve comprising hundreds of values. 

We therefore argue that the \emph{shape} of the curve, rather than its individual points, constitute the principal modeling target. In industrial processes, repeated executions of the same activity typically exhibit similar, recurrent shapes. So what varies between executions is not the fine structure of the curve itself, but a small set of characteristics, such as the timing of a transition, the duration of a phase, or its magnitude. Since the fine structure is already contained in historical curves, it does not need to be predicted point-wise–only these characteristics require estimation.

The starting point is the sensor data. Sensor measurements are typically recorded as continuous time series of timestamps and values at regular intervals (e.g., once per second), as shown in the first columns of the table on the left of Figure~\ref{fig:energy_model_extraction}. To extract activity-specific energy profiles, these measurements must be linked to process information by the corresponding activity and case\_id. This linkage is achieved either directly, when process monitoring systems (e.g., SCADA) store sensor and production data together, or indirectly through timestamps: sensor data recorded separately is merged with process data from other systems (e.g., MES or ERP) by assigning to each measurement the case\_id and activity active at that time.

The resulting profiles show recurring patterns per process activity, but the executions are not identical: values, peaks, and timing vary with the activity's duration, the product characteristics, the preceding activities, and external factors. In the infusion-bag example, the first batch of the day may require additional heating to warm up the machine, larger bags may need more sterilization heat, and preparation steps may be skipped when several orders of the same product run consecutively. Figure~\ref{fig:energy_model_extraction} shows two such executions of the same activity: both follow the same rapid increase, plateau, and decrease, albeit at different lengths and magnitudes. A direct point-wise comparison or modeling between such curves is therefore not appropriate. Instead, we propose that the extraction should capture the recurring shape once and, per execution, the characteristics that vary: when the phases occur (the breakpoints), how long they last (the segment durations), and how high the curve runs in each (the levels).

This leads to the extraction workflow of Figure~\ref{fig:energy_model_extraction}: the energy data is split into one profile per activity execution, and a reference curve is selected per activity and sensor. The reference is segmented once by change-point detection, and its values and breakpoints are stored for later reconstruction. The breakpoints are then aligned to every individual curve, and from each one the targets are read off: the duration fraction $\phi_m$ and the mean level $\lambda_m$ of every segment. Together with the case attributes, the external factors,  and the activity duration, these form the prepared energy data, on which one ML algorithm is trained per target.

\paragraph{Energy model extraction algorithm.} Building on the reasoning about the shape, we propose the Algorithm~\ref{algorithm:energy_model_extarction} to extract the energy models and the reference segments for every activity and sensor.

\textbf{Step A: Extract activity-level energy profiles.} For each combination of case\_id and activity, the energy signal is split into individual energy profiles, or curves $y_i$.

\begin{algorithm}[H]
\caption{Energy profile extraction.}
\label{algorithm:energy_model_extarction}
\begin{algorithmic}[1]
\Require Table $\mathcal{D}$ with case, activity, timestamps, attributes $\mathbf{x}^{\mathrm{attr}}$, external features $\mathbf{x}^{\mathrm{ef}}$ and energy values; ML algorithms $\mathcal{R}$; maximum number of segments $M_{\max}$

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
    \State Add to $\mathcal{E}$ one row with the features $(\mathbf{z}_i,\, a,\, |y_i|)$ and the $2M$ targets, summarizing each external-factor series by its mean over the window of the activity
\EndFor
\State Encode the categorical variables and keep the numerical variables unchanged

\State \textbf{(E) Train the segment models}
\For{each segment $m = 1,\dots,M$}
    \State Fit on $\mathcal{E}$ one ML algorithm $h \in \mathcal{R}$ for the duration $\phi_m$ and one for the level $\lambda_m$, under an absolute-error loss
    \State Keep each ML model only if it beats its constant fallback — the training mean of $\phi_m$ and the training median of $\lambda_m$ — on held-out curves
\EndFor
\State \Return the curve model $g$, the reference curve $r$ and the breakpoints $b$, for each sensor, and activity
\end{algorithmic}
\end{algorithm}

\begin{figure*}[H]
    \centering
    \includegraphics[width=1\textwidth]{5_step_dtw_method.pdf}
    \caption{Energy profile extraction. (a) Original curves with different lengths, phase timings and levels; the reference curve $r$ is the DTW medoid, a real execution; (b) The reference is segmented once by change-point detection into $M$ segments; (c) The breakpoints are carried through the DTW warping path onto every curve, giving the per-execution targets—segment duration fractions $\phi_m$ and levels $\lambda_m$—shown for one training curve; (d) A new execution rebuilt from its predicted durations and levels: the reference is warped onto the predicted segment durations and modulated by the level gains interpolated between the segment midpoints; the reference's transitions and within-phase texture survive, and the reconstruction adds no jumps of its own.}
    \label{fig:step_dtw_method}
\end{figure*}

\begin{figure*}[H]
    \centering
    \includegraphics[width=0.7\textwidth]{6_simulation.pdf}
    \caption{Process and energy simulation.}
    \label{fig:simulation}
\end{figure*}



\textbf{Step B: Select the reference curve.} Every curve is resampled to the median length $S$ and divided by its mean value, yielding its shape. The reference curve $r$ is set to the Dynamic Time Warping (DTW) medoid \citep{giorgino_computing_2009}: the curve whose shape has the smallest total DTW distance to the shapes of all others. The reference is deliberately a real measured execution rather than an average, since averaging would smooth out exactly the sharp transitions the segments are meant to preserve.

\textbf{Step C: Segment the reference curve.} The reference curve is segmented once by change-point detection: dynamic programming computes the best piecewise-constant approximation of its shape of $r$ for every number of segments up to $M_{\max}$, and the number of segments $M$ is selected via the Bayesian Information Criterion (BIC), a model-selection score that trades goodness of fit against the number of parameters, so every additional breakpoint has to pay for its two parameters. 

\textbf{Step D: Carry the segments onto every curve.} DTW carries the reference's breakpoints onto every training curve's own timeline. Since DTW compares time series with similar shapes but different temporal progressions by allowing local stretching and compression of the time axis, the warping path between each curve and $r$ matches corresponding phases even when they occur earlier, later, faster, or slower. From this alignment, the targets are read off: the duration fraction $\phi_{i,m}$ that execution $i$ spends in segment $m$, and the mean energy level $\lambda_{i,m}$ within it.

\textbf{Step E: Train the segment models.} One ML algorithm is fitted per target under an absolute-error loss. Each model is kept only if it beats a simple constant fallback—the mean duration or the median level of that segment—on held-out curves; otherwise the constant is used. An unpredictable parameter therefore degrades to the typical value of the activity, never to noise. 

Figure~\ref{fig:step_dtw_method} visualizes this pipeline in its entirety, from the raw executions and their reference curve to a profile reconstructed exclusively from predicted durations and levels. In summary, the proposed method predicts a small set of interpretable parameters—the duration and level of each phase—while the remaining structure is provided by a real measured curve. In the evaluation, this method is referred to as \textit{Step DTW}.

\subsection{Analysis}

In this section, we focus in greater detail on the analysis component, specifically on the simulation and the decision support capabilities that the Digital Twin and its simulation provide.

Figure~\ref{fig:simulation} shows the simulation stage, in which the Digital Twin generates synthetic process and energy data. The production plan and the expected external factors are provided as input to the process simulation, which unfolds each planned case into activities with simulated start times and durations. Each simulated activity instance is then combined with the external factors of its time window, from which the segment models predict its duration fractions $\phi$ and levels $\lambda$. The energy profile is then reconstructed from these predictions and the stored reference curves, at the corresponding simulated duration.



\begin{algorithm}
\caption{Simulation of the process and energy.}
\label{alg:simulation}
\begin{algorithmic}[1]
\Require Stochastic Petri net $\mathcal{N} = (N, M_0, M_f, \mathcal{G}, w)$ with $R_a$, $W_a$ and $\hat{f}_B$ (Algorithm~\ref{alg:pm_extraction_ml}); segment models $g$, reference curve $r$ and breakpoints $b$ per activity and sensor (Algorithm~\ref{algorithm:energy_model_extarction}); production plan $P$ with case attributes $\mathbf{x}^{\mathrm{attr}}_c$ and external factors $\mathbf{x}^{\mathrm{ef}}$; exit weights $\alpha < 1 < \beta$; step limit $k_{\max}$

\State \textbf{(A) Initialize the case}
\State Read case $c$ from $P$ and predict its time budget $B^\star = \hat{f}_B(\mathbf{x}^{\mathrm{attr}}_c)$
\State Set $M \gets M_0$, $\Delta \gets 0$, $k \gets 0$, $\nu_a \gets 0$; draw the repeat quotas $\kappa_a \sim R_a$

\State \textbf{(B) Replay the Petri net}
\While{$M \neq M_f$, $\Delta < B^\star$ and $k < k_{\max}$}
    \State Weight the enabled transitions by $w$, discounting activities at quota ($\nu_a \geq \kappa_a$) and scaling by $\alpha$ the transitions leading to $M_f$
    \State Sample $t^\star$ proportionally to the weights and fire it, $M \gets \mathrm{fire}(M, t^\star)$
    \If{$\ell(t^\star) = a$ is a visible activity}
        \State Obtain the duration $d^\star$ from $D_{t^\star} \in \mathcal{G}$ and draw the idle time $\delta \sim W_a$
        \State Append $(a,\, \Delta + \delta,\, d^\star)$ to $\mathcal{L}$; $\Delta \gets \Delta + \delta + d^\star$; $\nu_a \gets \nu_a + 1$
    \EndIf
    \State $k \gets k+1$
\EndWhile
\State Scale by $\beta$ the transitions leading to $M_f$ and replay until $M = M_f$

\State \textbf{(C) Predict and place the energy profiles}
\For{each activity instance $(a, \Delta_a, d^\star) \in \mathcal{L}$ and each sensor}
    \State Assemble $\mathbf{z}^\star_a$: $\mathbf{x}^{\mathrm{attr}}_c$, the external factors averaged over $[\Delta_a,\, \Delta_a + d^\star]$, and $d^\star$
    \State Predict the duration fractions $\phi^\star_m$ and levels $\lambda^\star_m$ with the segment models $g$ of the activity and sensor
    \State Reconstruct $\hat{y}^\star$ at length $d^\star$: boundaries at the cumulative $\phi^\star$ scaled to $d^\star$, $r$ warped segment-wise onto them, level gains interpolated between the segment midpoints
    \State Insert $\hat{y}^\star$ into the energy timeline $\hat{Y}$ over $[\Delta_a,\, \Delta_a + d^\star]$
\EndFor

\State \Return $\mathcal{L}$ and $\hat{Y}$, repeating (A)--(C) per case of $P$ and superposing the timelines
\end{algorithmic}
\end{algorithm}

\paragraph{Simulation of the process and energy.} Algorithm~\ref{alg:simulation} details how the process and energy models introduced above are used to generate synthetic process and energy data for a given production plan.

\textbf{Step A: Initialize the case.} The simulation processes one case of the production plan and predicts its time budget, i.e., how long the case is expected to take in total. A repeat quota is sampled for every activity from the repetitions observed in the training log, so that loops in the net produce realistic repetitions rather than an unbounded random choice at every visit.

\textbf{Step B: Replay the Petri net.} At each step, the enabled transitions are weighted with the routing probabilities of the process model, subject to two corrections: activities that have reached their quota are strongly discounted but not blocked, and transitions that would terminate the case are discounted while the budget remains unspent. A transition is then sampled and fired. For an activity, its duration is predicted by the duration model of the corresponding transition, the preceding idle time is drawn from its waiting distribution, and the instance is appended to the simulated log; both operations consume the budget. Once the budget is spent, the ending transitions are favored instead, ensuring that the case closes on a structurally valid end.

\textbf{Step C: Predict and place the energy profiles.} Each simulated instance is joined with the external factors of its time window, consistent with the summary used at training time. On this basis, the segment models predict the duration fractions and levels, and the profile is reconstructed using the stored reference curve and placed on the timeline over the interval of the instance.

Steps~(A)–(C) are repeated for every case and the profiles are superposed into a single energy timeline: a synthetic event log with realistic sequences and durations, together with its corresponding energy behavior, derived solely from a production plan and the expected external factors.


\paragraph{Analytics.} By linking energy behavior to the process events that cause it, and by enabling forward simulation of planned scenarios, the proposed Process and Energy Digital Twin supports decision-making across four analytical levels: descriptive, diagnostic, predictive, and prescriptive. Table~\ref{table:analytics_levels} summarizes representative analyses examples enabled at each level.


\begin{table}[H]
\caption{Example of analyses offered by the Process and Energy Digital Twin.}
\label{table:analytics_levels}
\begin{tabular}{p{1.4cm}p{6cm}}
\toprule
Type & Examples \\
\midrule
Descriptive &
$\bullet$ Reconstruction and visualization of process-linked energy profiles \newline
$\bullet$ Assignment of energy consumption to activities, cases, machines, and products \newline
$\bullet$ Characterization of historical energy behavior
\\[0.5em]
Diagnostic &
$\bullet$ Explanation of why energy profiles differ across executions \newline
$\bullet$ Identification of the influence of duration, product, disturbances, and operational conditions on the process and energy behavior \newline
$\bullet$ Detection of abnormal relationships between process execution and energy behavior
\\[0.5em]
Predictive &
$\bullet$ What-if analyis and simulation of process execution and the resulting energy profiles \newline
$\bullet$ Estimation of the energy impact of different production scenarios
\\[0.5em]
Prescriptive &
$\bullet$ Recommendation of process and planning decisions that improve energy efficiency \newline
$\bullet$ Optimization of schedules and operating conditions based on predicted energy outcomes \newline
$\bullet$ Support for energy-aware process design and control
\\
\bottomrule
\end{tabular}
\end{table}


\section{Evaluation}
\label{sec:evaluation}

In order to test whether the proposed approach for process and energy modeling accurately represents and simulates the real-world behavior of the system, we employ a structured evaluation protocol, together with several benchmark algorithms and datasets.

\subsection{Evaluation protocol}

Figure~\ref{fig:evaluation_protocol} illustrates the evaluation protocol. We first apply a temporal split to the data, allocating 70\% for training and 30\% for testing. Model extraction is performed on the training data, corresponding to the earlier portion of the timeline, while the evaluation metrics are computed on the test data, corresponding to the later portion. This setup prevents temporal leakage and ensures that the models are evaluated on unseen, future data.

We propose three evaluations to properly assess the individual components of the proposed framework with empirical evidence: the process model, the individual energy profiles, and the combined process and energy behavior. The results are presented accordingly, with the evaluation of the combined process and energy behavior complemented by a visualization of the simulated outcomes. Each evaluation is detailed in the following subsections, together with its corresponding metrics.

\begin{center}
    \includegraphics[width=1\columnwidth]{7_evaluation_protocol.pdf}
    \captionof{figure}{Evaluation protocol.}
    \label{fig:evaluation_protocol}
\end{center}


\subsection{Evaluation 1: Process Models}
\label{sec:complete-curve-eval}

We first evaluate whether the proposed methods are able to reproduce the structure of the process. Three process models are considered, each represented as a Petri net: \textit{Alpha Petri net}, the Alpha miner used as a baseline; \textit{Best Petri net}, the best discovered Petri net; and \textit{Best Petri net + Budget}, the best Petri net combined with duration budgeting at simulation time. Each of them is crossed with the three duration predictors: \textit{baseline}, which samples from the fitted statistical distributions; \textit{ml\_local}, one ML model per activity; and \textit{ml\_global}, a single model shared across all activities—resulting in nine model combinations in total.

We first assess the structural conformance of the three process models—\textit{Alpha Petri net}, \textit{Best Petri net}, and \textit{Best Petri net + Budget}—using four process conformance metrics \citep{buijs_quality_2014}, obtained by replaying the real test log on each corresponding Petri net: \textbf{Fitness} measures how completely the model reproduces the observed behavior, penalizing missing and remaining tokens during replay \citep{rozinat_conformance_2008}. \textbf{Precision} penalizes models that enable activities never observed at that point in the log \citep{munoz-gama_fresh_2010}. \textbf{Generalization} rewards models that remain open to plausible but unobserved behavior rather than overfitting to the training log. \textbf{Simplicity} penalizes nets with a large number of places, transitions, and arcs. Fitness results from replaying individual traces and thus yields one value per test case, whereas the other three metrics are defined on the model as a whole and yield a single value per process.

Beyond structural conformance, we evaluate the temporal and quantitative fidelity of the simulated log. The \textbf{Evt-Ratio Error} captures the ratio between the number of simulated and real events. The \textbf{Activity duration}, i.e., the duration of individual activities, and the \textbf{Lead time}, i.e., the total elapsed time of a case from start to end, are both measured with the Weighted Absolute Percentage Error (\textbf{WAPE}). The WAPE is a relative metric that handles small values better than the commonly used Mean Absolute Percentage Error \citep{hewamalage_forecast_2023} and, being scale-free, allows comparison across processes whose durations differ by orders of magnitude. The activity-duration WAPE is computed within each case and reported as the median across the test cases, while the lead-time WAPE is pooled over all test cases.

\subsection{Evaluation 2: Individual Energy Profiles}

This evaluation assesses the effectiveness of the proposed energy-modeling method at the level of \emph{individual} curves per activity, i.e, how accurately the proposed profile models reproduce individual curves compared to alternative methods. For this purpose, we use the process cases of the real test set–together with their activities, relevant attributes, and external factors such as weather variables and the day of the week–to predict energy profiles with each of the proposed approaches. The predicted curves are then compared against the corresponding ground-truth profiles of the test set.

The comparison is complemented by an ablation study in which individual components of the methodology are removed, allowing the contribution of each component to be quantified and the underlying design decisions to be supported not only by the literature but also by empirical evidence. The full proposed method, \textit{ML Step DTW} (Algorithm~\ref{algorithm:energy_model_extarction}), predicts DTW-transferred segments whose durations and levels are estimated by ML algorithms conditioned on the external factors. The ablations each remove one component at a time: \textit{ML DTW (no steps)} replaces the segment parameterization with a per-position regression on the DTW-aligned curves, and \textit{ML (no DTW)} omits the DTW alignment entirely. 

As an alternative model class also represented in the literature, we include two sequence-to-sequence variants: \textit{Seq2Seq DTW (aligned)}, which replaces the regression with a sequence-to-sequence model trained on the DTW-aligned curves, and \textit{Seq2Seq (DTW-scored)}, which follows \citet{worrlein_using_2024} and trains on the raw curves while selecting the model by scoring its generated curves against a DTW reference. We further include \textit{Median per Activity \& Sensor} as a baseline, adapted from the evaluation of \citet{gonzalez_process_2025}: for each combination of activity and sensor, it predicts a constant profile at the median power level observed in the training data. It thus conditions only on the activity itself, ignoring the shape of the demand and all other input factors.

Since the goal is to assess how realistically the predicted profiles reproduce the behavior of the real ones rather than their point-wise accuracy, the quality of the predictions is measured with five statistical metrics, each computed on the predicted and the ground-truth curve and compared between the two: \textbf{Sum}, the summed value of the curve, corresponding to total energy for demand sensors; \textbf{Max}, the peak value of the curve; \textbf{Mean} and \textbf{Std}, the mean and the standard deviation of the curve \citep{wang_generating_2020}; and \textbf{Roughness}, the mean absolute difference between consecutive samples, which quantifies the point-to-point variability of the curve and thus penalizes predictions that are unrealistically smooth \citep{christ_time_2018}. The five metrics are further aggregated into a single \textbf{Overall realism} score, summarizing the realism of a predicted profile in one value.

Each deviation is normalized by the typical magnitude of the corresponding property over the real test curves of the respective sensor, $|f_{\mathrm{pred}} - f_{\mathrm{real}}| \,/\, \overline{|f_{\mathrm{real}}|}$, rendering the metrics unitless and comparable across sensors of different physical scales.

\subsection{Evaluation 3: Complete Process and Energy}

This evaluation combines the process models from the process evaluation–\textit{Alpha Petri net}, \textit{Best Petri net} and \textit{Best Petri net + Budget}–each paired with the best-performing duration predictor from the process evaluation and the best-performing curve model from the energy evaluation. 

The evaluation is carried out per case\_id, i.e., per order of the production schedule case, and per sensor, comparing the \emph{complete} energy profile of each test case with its simulated counterpart, rather than isolated per-activity curves. To this end, we simulate the process based on the production plan and external factors, and use the simulated process data to predict the energy profiles with the energy models. We then concatenate all these profiles into the complete energy profile of each test case. We apply the same metrics used in the previous evaluation to the complete curves.

To assess how much the process model contributes to the quality of the energy prediction, we further include two approaches that do not rely on it: the \textit{Median per Sensor (no process model)}, which reuses the median energy curve of the sensor, and a \textit{Profile-generator (no process model)}, which generates the energy profile of a case directly from a learned case-level energy curve, bypassing both process modeling and simulation.

\subsection{Algorithms for the implementation}

The methods and experiments are implemented in Python. For the ML algorithms used to predict activity durations for process modeling and to model the energy profiles, several candidate algorithms compete per prediction target, and the best cross-validated one is retained: standard implementations of Linear Regression, Ridge Regression, Huber Regression, Random Forest, Multilayer Perceptron, and Histogram-based Gradient Boosting from the scikit-learn library \citep{pedregosa_scikit-learn_2011}, as well as Extreme Gradient Boosting from the XGBoost library \citep{chen_xgboost_2016}. The segment duration and level models of Algorithm~\ref{algorithm:energy_model_extarction} are trained under an absolute-error loss. For curve prediction specifically, we additionally include two sequence-to-sequence algorithms based on a Long Short-Term Memory (LSTM) neural network implemented in PyTorch \citep{paszke_pytorch_2019}. Hyperparameter tuning of the ML algorithms is performed using Optuna \citep{ozaki_optunahub_2025}.

For process modeling, we use three process discovery algorithms—the Alpha Miner \citep{van_der_aalst_workflow_2004}, the Heuristic Miner \citep{weijters_flexible_2011, weijters_process_2006}, and the Inductive Miner \citep{leemans_discovering_2013}—as implemented in the PM4Py library \citep{berti_pm4py_2023}. For curve modeling, we use the Dynamic Time Warping (DTW) algorithm \citep{giorgino_computing_2009}, and distribution fitting (Normal, Lognormal, Exponential, Gamma) is performed with the SciPy library \citep{virtanen_scipy_2020}.

\subsection{Datasets}

For the evaluation, we use event logs and energy time series from one synthetic and five real-world industrial processes, summarized in Table~\ref{table:datasets}.

\begin{table}[width=.9\linewidth,cols=6,pos=h]
\caption{Dataset information overview. Hours of operation refer to the recorded length of the energy time series; span is the calendar period covered.}
\label{table:datasets}
\setlength{\tabcolsep}{3pt}
\begin{tabular}{@{}lrrrrr@{}}
\toprule
dataset & \makecell{number\\of cases} & \makecell{number of\\different\\activities} & \makecell{number\\of sensors} & \makecell{hours of\\operation} & \makecell{span\\(days)} \\
\midrule
process\_1 & 300 & 25 & 6 & 604 & 50 \\
process\_2 & 48 & 12 & 5 & 512 & 21 \\
process\_3 & 49 & 12 & 5 & 512 & 21 \\
process\_4.1 & 160 & 5 & 18 & 2387 & 212 \\
process\_4.2 & 156 & 5 & 17 & 2463 & 205 \\
process\_5 & 53 & 20 & 28 & 501 & 29 \\
\midrule
total & 766 & 79 & 79 & 6979 & 538 \\
\bottomrule
\end{tabular}
\end{table}

\textit{Process\_1} is a synthetic dataset of medical infusion-bag production, covering distillation, bottling, autoclave sterilization, and packaging. It is constructed realistically: events are generated by a DES that follows the real production logic of such a line, and the electricity, steam, and cooling profiles are derived from thermodynamic calculations of the process equipment, such that the ground-truth link between activities and energy behavior is known (Appendix~\ref{app:process1_sim}).

\textit{Process\_2} and \textit{process\_3} are the two production lines of a wet-mixing area at a baby-food manufacturer, where milk is heated, homogenized, and cooled. The lines run independently but share the steam meter of the mixing process.

\textit{Process\_4.1} and \textit{process\_4.2} come from the same spray-drying tower producing milk powder, dominated by steam-based air heating. They cover the periods before and after a modification of the tower, which changed run lengths, interruption frequency, and the level of several process variables; we therefore treat them as two distinct processes.

Finally, \textit{process\_5} is the heating network of a pasteurization plant at a fruit-juice producer, with steam-based heating and chilled-water cooling. 

Across all real processes, measurements include thermal heating and cooling power, process temperatures, product recipe data, and event logs. Weather variables—ambient temperature, relative humidity, and solar irradiance—are incorporated as external features for energy modeling, while day-of-week and month serve as external features for process and energy modeling.

Collectively, the datasets comprise 766 cases and 79 energy sensors covering close to 7,000 hours of operation, providing a diverse and realistic basis for evaluating the proposed methods across varying process and energy characteristics.

The synthetic \textit{process\_1} and the code repository allow the complete reproducibility of the results: the dataset can be regenerated with the provided simulation, and all methods and evaluations can be executed on it end to end. The data of the remaining processes originates from real production environments and is confidential, and therefore cannot be shared. The code, the simulation of \textit{process\_1}, and the complete results are available in the online repository\footnote{Repository \url{XXX-REPOSITORY-LINK}}.

\section{Results}
\label{sec:results}

\subsection{Results of Evaluation 1: Process Models}

In total, the evaluation produces 9{,}192 individual results, one per process, method, and case: 6{,}420 on the training cases and 2{,}772 on the test cases, on which the reported metrics are calculated. We report aggregated medians to avoid that the results are dominated by a few extreme values; the complete individual results are available in the repository.

Table~\ref{tab:process_results} presents the results of the first evaluation. \textbf{Precision}, \textbf{Generalization}, and \textbf{Simplicity} are model-level metrics with no per-case counterpart and are therefore reported as the median across processes. \textbf{Fitness}, in contrast, is computed per test case and reported as the median over all pooled test cases, and the timing metrics are likewise pooled over all test cases. 

It can be seen that the \textit{Best Petri net} and \textit{Best Petri net + Budget} variants perform well across all four metrics, achieving a perfect \textbf{Fitness} of 1.000 and all other metrics of at least 0.66. As both variants use the same underlying process model, their results are identical across these metrics. By comparison, the \textit{Alpha (Baseline)} performs on average 25\% worse across these metrics.

%Table~\ref{tab:process_results} presents the results of the first evaluation: the discovery metrics are reported as the median across processes, and the timing metrics are pooled over all test cases. The \textit{Best Petri net} and \textit{Best Petri net + Budget} variants perform well across \textbf{Fitness}, \textbf{Precision}, \textbf{Generalization}, and \textbf{Simplicity}, achieving a near-perfect \textbf{Fitness} of 0.994 and all other metrics above 0.66. As both variants use the same underlying process model, their results are identical across these metrics. By comparison, the \textit{Alpha (Baseline)} performs on average 31\% worse across these metrics.

Regarding the duration metrics, approaches using \textit{ml\_local} predict the \textbf{Activity duration} more accurately, outperforming \textit{ml\_global} by 12\% on average and the baseline durations by 42\%. For this metric, \textit{Best Petri net} and \textit{Best Petri net + Budget} do not differ substantially, with \textit{Best Petri net} combined with \textit{ml\_local} achieving the best result (21.1\% \textbf{WAPE}).

For lead times, however, the differences between the methods are considerably large. \textit{Best Petri net + Budget} achieves considerably better results than the other methods, outperforming the best configuration of \textit{Best Petri net} by approximately 61\% and that of \textit{Alpha (Baseline)} by 60\%. A similar pattern holds for the \textbf{Evt-Ratio Error}, where \textit{Best Petri net + Budget} outperforms \textit{Best Petri net} by 25\%. On these lead-time metrics, \textit{Best Petri net} and \textit{Alpha (Baseline)} perform poorly, with a \textbf{Lead time WAPE} above 57\%.

The process model is computationally negligible: as shown in Appendix~\ref{app:computational_time}, discovery and duration-model fitting take seconds per process, and while \textit{ml\_local} trains almost five times slower than \textit{ml\_global}, since it fits one model per activity, it is the cheaper of the two at simulation time.

In summary, \textit{Best Petri net + Budget} combined with \textit{ml\_local} achieves the best overall results, combining the highest process discovery quality with the lowest \textbf{Evt-Ratio Error} and \textbf{Lead time WAPE} (on par with \textit{ml\_global}). Across all methods, \textit{ml\_local} is the strongest duration prediction approach, outperforming \textit{ml\_global} and the baseline durations.

\begin{table*}[t]
\centering
\caption{Process modeling and timing accuracy (test set results)}
\label{tab:process_results}
\vspace{-0.5em}
\setlength{\tabcolsep}{8pt}
\renewcommand{\arraystretch}{1.3}
\footnotesize
\begin{tabular}{llccccccc}
\toprule
Method & \makecell[l]{Time\\approach} & Fitness & Precision & \makecell{Generalization} & Simplicity & \makecell{Evt-Ratio\\Error} & \makecell{Activity\\duration\\WAPE (\%)} & \makecell{Lead time\\WAPE (\%)} \\
\midrule
\multirow{3}{*}{\makecell[c]{Alpha \\ Petri net \\ (Baseline)}} & baseline & 0.827 & 0.400 & 0.586 & 0.489 & 0.333 & 39.440 & 68.311 \\
 & ml\_global & 0.827 & 0.400 & 0.586 & 0.489 & 0.333 & 27.375 & 57.677 \\
 & ml\_local & 0.827 & 0.400 & 0.586 & 0.489 & 0.333 & 25.795 & 58.033 \\
\cline{1-9}
\multirow{3}{*}{\makecell[c]{Best \\ Petri net}} & baseline & \textbf{1.000} & \textbf{0.728} & \textbf{0.664} & \textbf{0.660} & 0.333 & 43.875 & 74.272 \\
 & ml\_global & \textbf{1.000} & \textbf{0.728} & \textbf{0.664} & \textbf{0.660} & 0.333 & 25.709 & 59.853 \\
 & ml\_local & \textbf{1.000} & \textbf{0.728} & \textbf{0.664} & \textbf{0.660} & 0.333 & \textbf{21.139} & 59.252 \\
\cline{1-9}
\multirow{3}{*}{\makecell[c]{Best \\ Petri net \\ + Budget}} & baseline & \textbf{1.000} & \textbf{0.728} & \textbf{0.664} & \textbf{0.660} & 0.357 & 38.399 & 35.057 \\
 & ml\_global & \textbf{1.000} & \textbf{0.728} & \textbf{0.664} & \textbf{0.660} & 0.267 & 26.203 & \textbf{23.011} \\
 & ml\_local & \textbf{1.000} & \textbf{0.728} & \textbf{0.664} & \textbf{0.660} & \textbf{0.250} & 22.630 & 23.165 \\
\cline{1-9}
\bottomrule
\end{tabular}
\vspace{0.5em}
\noindent\raggedright\footnotesize

Process models: Alpha Petri net (baseline miner), Best Petri Net (best discovered net) and Best Petri Net + Budget (adds duration budgeting at simulation time), each crossed with three activity-duration predictors: baseline samples the fitted statistical distributions, ml\_global is one ML algorithm for all activities, ml\_local one per activity. Fitness, Precision, Generalization and Simplicity score discovery quality (higher is better; median across processes). Evt-Ratio Error is the simulated-to-real event-count ratio error (lower is better). Activity duration and Lead time are WAPEs against the real test cases (lower is better): the former the median over cases, the latter pooled over them. \textbf{Bold} = best per column.
\end{table*}

\subsection{Results of the Evaluation 2: Individual Energy Profiles}

The results are aggregated using the median across all methods, sensors, processes, and activities. In total, the second evaluation comprises 407{,}798 individual results, one per approach and predicted profile instance: 272{,}325 on the training instances and 135{,}473 on the test instances, on which the reported metrics are calculated. The complete individual results are available in the code repository.

Table~\ref{tab:individual_profile_realism} presents the results. They are sorted by the overall realism value. \textit{ML Step DTW} performs best overall, achieving an overall realism error of 0.189 and the lowest errors in \textbf{Max}, \textbf{Std}, and \textbf{Roughness}. It is followed by \textit{ML DTW (no steps)} (17\% worse overall) and \textit{ML (no DTW)} (21\% worse overall). The advantage of the proposed method lies in the dynamics: its \textbf{Std} and \textbf{Roughness} errors are clearly the lowest, indicating that its curves reproduce the amplitude and the point-to-point variability of the real profiles rather than smoothing them away. Removing further components degrades the realism consistently, as the ablated variants confirm. The sequence-to-sequence variants performed worse than all regression-based methods, with \textit{Seq2Seq (DTW-scored)} and \textit{Seq2Seq DTW (aligned)} performing 41\% and 43\% worse overall than the best method, respectively. The \textit{Median per Activity \& Sensor} baseline matches the aggregate quantities \textbf{Sum} and \textbf{Mean} slightly better, as a constant at the typical power level of an activity is a robust estimate of these properties. However, being constant, it cannot reproduce any temporal dynamics, which yields the worst \textbf{Std} and \textbf{Roughness} errors and places it last overall (46\% worse than the best method).

Appendix~\ref{app:Individual_profiles} shows individual energy profiles from the test set, comparing the real curves of single activity executions with the predictions of all methods. The examples illustrate the range of prediction quality: some profiles are volatile and difficult to model, some predictions are shifted in time but reproduce the overall form and magnitude of the real curve, and others match it almost point-wise.

This realism also comes at a moderate computational cost: as shown in Appendix~\ref{app:computational_time}, \textit{ML Step DTW} predicts a curve roughly twice as fast as the Seq2Seq methods and trains far cheaper than the ML variants without step segmentation.

In summary, it can be said that the \textit{ML Step DTW} achieves the most realistic individual energy profiles, with the best \textbf{Overall} value and the lowest errors in the dynamic properties \textbf{Std} and \textbf{Roughness}. Each of its components contributes to this result, as removing them degrades realism consistently.



\begin{table*}[H]
\centering

\begin{minipage}{16cm}
\centering

\captionsetup{
    justification=centering,
    singlelinecheck=false,
    format=plain
}

\caption{Curve realism for individual energy-profile prediction.}
\label{tab:individual_profile_realism}

\vspace{-0.5em}

\begin{tabular}{p{6cm}cccccc}
\toprule
Method & Sum & Max & Mean & Std & Roughness & Overall \\
\midrule
ML Step DTW (proposed) & 0.015 & \textbf{0.064} & 0.044 & \textbf{0.354} & \textbf{0.470} & \textbf{0.189} \\
ML DTW (no steps) & 0.015 & 0.073 & 0.047 & 0.431 & 0.540 & 0.221 \\
ML (no DTW) & 0.015 & 0.079 & 0.048 & 0.446 & 0.556 & 0.229 \\
Seq2Seq (DTW-scored) & 0.018 & 0.085 & 0.061 & 0.528 & 0.646 & 0.267 \\
Seq2Seq DTW (aligned) & 0.016 & 0.081 & 0.057 & 0.528 & 0.665 & 0.270 \\
Median per Activity \& Sensor (baseline) & \textbf{0.012} & 0.078 & \textbf{0.041} & 0.536 & 0.712 & 0.276 \\
\bottomrule
\end{tabular}

\vspace{0.5em}

\parbox{16cm}{%
\footnotesize
Curve realism for individual profile prediction, TEST set. Each column is the median normalized absolute error of that curve property, so 0 is perfect and lower is better; Overall is their average.
\textbf{Bold} marks the best value per column.
}

\end{minipage}

\end{table*}


\begin{table*}[H]
\centering

\begin{minipage}{\textwidth}
\centering

\captionsetup{
    justification=centering,
    singlelinecheck=false,
    format=plain
}

\caption{Complete energy-profile comparison, all processes.}
\label{tab:energy_profile}

\vspace{-0.5em}

\small
\begin{tabularx}{\linewidth}{>{\hsize=1.3\hsize\linewidth=\hsize\raggedright\arraybackslash}X>{\hsize=0.7\hsize\linewidth=\hsize\raggedright\arraybackslash}Xcccccc}
\toprule
Method (process model) & Curve generation & Sum & Max & Mean & Std & Roughness & Overall \\
\midrule
Best Petri net + Budget + ml\_local & Step DTW & \textbf{0.211} & \textbf{0.074} & 0.077 & \textbf{0.372} & \textbf{0.602} & \textbf{0.267} \\
Best Petri net + ml\_local & Step DTW & 0.379 & 0.077 & \textbf{0.070} & 0.385 & 0.605 & 0.303 \\
Alpha Petri net + ml\_local & Step DTW & 0.431 & 0.083 & 0.076 & 0.447 & 0.625 & 0.332 \\
\midrule
Profile-generator (no process model) & Stochastic generator & 0.348 & 0.197 & 0.092 & 1.280 & 9.980 & 2.379 \\
\midrule
Median per Sensor (no process model) & Median per Sensor & 0.320 & 0.165 & 0.088 & 0.908 & 0.942 & 0.485 \\
\bottomrule
\end{tabularx}

\vspace{0.5em}

\parbox{\linewidth}{%
\footnotesize
\textit{Median per Sensor (baseline)}: one median level per sensor, pooled over all its activities, emitted as a flat line for every case. \textit{Alpha Petri net}: net discovered by the alpha miner. \textit{Best Petri net}: best discovered net per process, selected on the training split by the mean of Fitness, Precision, Generalization and Simplicity. \textit{Best Petri net + Budget}: the same net, with each case generated to match its predicted total-duration budget. \textit{Profile-generator}: stochastic profile generator. The three Petri-net rows use the \textit{Step DTW} curve predictor of Evaluation~2. Each row names the process model the cases are generated from, including its duration predictor (ml\_local = one per activity), and \textit{Curve generation} is how the load curve of each activity or case is then produced. Cells are the median over (process, case, sensor) of the paired per-case relative error $|f(\mathrm{pred})-f(\mathrm{real})|/\overline{|f(\mathrm{real})|}$. Lower is better; \textbf{bold} = best per column. Overall is the average across the metric columns.
}

\end{minipage}

\end{table*}

\subsection{Results of Evaluation 3: Complete Process and Energy Profiles}

In this subsection we evaluate the complete pipeline: a simulated energy profile is generated per case and compared with the corresponding real test case. Based on the previous evaluations, the three process models (\textit{Alpha Petri net (Baseline)}, \textit{Best Petri net} and \textit{Best Petri net + Budget}) are combined with the best duration model (\textit{ml\_local}) to simulate the test production plan cases, and the profiles are predicted with the best method from the second evaluation (\textit{ML Step DTW}). They are compared against two process-agnostic approaches, the \textit{Profile-generator} and the \textit{Sensor median (baseline)}. In total, this evaluation comprises 54{,}195 individual results, one per method, process, case, sensor, and complete profile over the 231 simulated test cases; we report medians, as the skewed distributions would let a few extreme values dominate a mean, and the complete individual results are available in the repository.

Overall, \textit{Best Petri net + Budget} performs best, achieving the best values in all metrics except \textbf{Mean}, followed by \textit{Best Petri net} (13\% worse overall) and \textit{Alpha Petri net} (24\% worse overall). The approaches that do not consider the process performed clearly worse: \textit{Sensor median} is 82\% worse overall than the best method, and \textit{Profile-generator} yields by far the worst results, driven by a very large \textbf{Roughness} error.

Notably, the process-simulating methods produce similar metrics overall, differing substantially only in \textbf{Sum}, where \textit{Best Petri net} performs 80\% worse than \textit{Best Petri net + Budget}. This reflects the effect of the duration budgeting, which corrects the total case duration and, with it, the total energy.

In summary, \textit{Best Petri net + Budget} combined with \textit{ml\_local} and \textit{ML Step DTW} achieves the most realistic complete energy profiles, being the best in every metric except \textbf{Mean}. The methods that simulate the process clearly outperform the approaches that do not consider it.

\subsection{Visualization of the Process and Energy}

Figure~\ref{fig:total_profile} shows energy profiles of \textit{process\_1} cases for all methods considered in the complete-energy-profile evaluation. The process-based methods produce energy curves similar to the real one, reproducing the shape and magnitude of the demand peak, though occasionally shifted in time relative to the real case. The \textit{Alpha Petri net} additionally simulates activities that did not occur in the real case, at times placing the peak far from its real position. \textit{Best Petri net + Budget} produces the profiles closest to the real ones, with case lead times that are also the closest match–consistent with the low realism values shown next to each case. \textit{Best Petri net} produced similar profiles, as it shares the process model and activity duration prediction with \textit{Best Petri net + Budget}, but without duration budgeting, the simulated case durations are less accurate.

For the methods that do not use the process to construct the profiles, the resulting curves appear largely random, with peaks and spikes distributed across the case that do not follow the real process structure–even though their metrics shown in Table~\ref{tab:energy_profile} may not appear far from those of the process-based methods. This highlights the importance of visual inspection in addition to quantitative metrics.

Figure~\ref{fig:process_and_profile} shows an example from process \textit{process\_1}, with individual energy profiles of selected autoclaving activities: real test cases on the left and simulated cases on the right. 



\begin{figure*}[p]
    \centering
    \includegraphics[width=1\textwidth]{8_process_energy_visual.pdf}
    \caption{Autoclave cooling water demand. The sensor is the sum of all three autoclaves that can run in the simulation. Due to stochastic variation, not necessarily the same autoclave starts in all simulations and the test set.}
    \label{fig:total_profile}
\end{figure*}

\begin{figure*}[p]
    \centering
    \includegraphics[width=1\textwidth]{9_process_and_profiles.pdf}
    \caption{Process 1 heuristic graph with material flow and machine activities of the autoclave, with energy profiles of the test set and the simulated test set with the best method.}
    \label{fig:process_and_profile}
\end{figure*}



The similarity between real and simulated values is evident. The discovered net reproduces the structure and activity frequencies of the real process, and the simulated energy profiles per activity match the shape and magnitude of the real ones.

\section{Discussion}
\label{Discussion}

In this article, we proposed a framework for building Process and Energy Digital Twins in manufacturing. Using data-driven methods, the approach captures both the discrete process flow and the dynamic energy behavior of a production system, so that the two can be analyzed together and simulated for new production plans, supporting the understanding of how process execution generates energy demand and the improvement of process- and energy-related decisions.

As motivated in the introduction, existing Process Mining approaches typically model production processes \citep{camargo_discovering_2021, lugaresi_automated_2023, castiglione_automated_2024} without an integrated, continuous energy perspective; existing Digital Twin approaches for process and energy in manufacturing represent energy only through static, aggregated KPIs \citep{khodadadi_data-driven_2024, khodadadi_automated_2026}; and existing ML-based energy models predict aggregate consumption or time series from exogenous covariates alone, without grounding them in discrete process execution \citep{he_generic_2020, zhang_data-driven_2021, mawson_deep_2020}. Our framework closes this gap by jointly and causally modeling discrete process execution and continuous energy behavior, capturing how production schedules and external factors shape both. Extensive experiments using simulated and real-world event data and energy profiles from production environments validate our design decisions. In the following, we discuss our scientific contributions to these three literature streams, the practical implications of our findings.

\subsection{Scientific Contribution and Implications}
To realize the conceptual argument that energy modeling must be causal, we make five \emph{technical} contributions: (1) manufacturing-specific considerations for the automated extraction of process and energy models; (2) an algorithm for extracting process models with an integrated energy perspective; (3) an algorithm for extracting dynamic energy profiles; (4) an algorithm for coupling the two models for simulation, such that process and energy dynamics interact and constrain one another and thereby reflect the behavior of the real system; and (5) methods for visualization, analysis, and decision support built on the proposed models. In the following, we detail how these technical contributions advance the three literature streams underlying this work.

\vspace{\baselineskip}
\textbf{Contribution to the Process Mining literature: activity granularity and explicit time modeling are prerequisites for energy modeling.}

Process mining literature on data-driven modeling of industrial processes \citep{camargo_discovering_2021, friederich_data-driven_2022, lugaresi_automated_2023, castiglione_automated_2024, van_der_aalst_data_2016, van_der_aalst_process_2022} predominantly evaluates discovered process models with control-flow metrics and places its main emphasis on material flow. We contribute to this literature by showing that separating material flow from the underlying machine activities is essential once energy is the modeling target, and that temporal information—at the level of individual activity executions and of total case durations—is equally central to this representation.
 
Time matters at two levels: at the activity level, the duration of an activity instance directly shapes its individual energy profile; at the level of the complete simulation, activity durations and total case lead time determine when the profiles occur and how they compose into the overall energy profile of a case. Evaluation~1 shows that explicitly accounting for total case duration achieves a substantially better fit for case lead time than sampling durations from the fitted distributions of a stochastic Petri net. Within the same evaluation, the \textit{ml\_local} variant—one model per activity—outperforms \textit{ml\_global} on individual activity durations, plausibly because a dedicated model per activity faces a simpler learning problem than a single model covering the entire process. Evaluation~3 confirms the downstream relevance of this contribution: the budget-based method, which corrects the total case duration, performs best overall, and its predicted energy profiles provide the closest match to the test data. Process structure and duration modeling therefore propagate directly into energy profile quality—an argument for granular, time-aware process modeling that goes beyond what control-flow accuracy alone would motivate.

\vspace{\baselineskip}
\textbf{Contribution to the ML-based energy modeling literature: continuous profiles from discrete process activities.}

A second stream of literature models industrial energy behavior with machine learning, largely focusing either on predicting point values (e.g., regression models for aggregated consumption) or on modeling energy time series from temporal covariates alone \citep{he_generic_2020, zhang_data-driven_2021, mawson_deep_2020}. We extend this literature, and the closely related line of work that predicts energy curves from discrete simulations \citep{kouki_input_2017, woerrlein_method_2020, worrlein_using_2024}, by showing that a complete, continuous energy curve can be generated accurately from strictly discrete inputs—activity executions and product attributes—which is a considerably harder problem than the two settings above \citep{woerrlein_method_2020, worrlein_using_2024}.
 
Our results show that this is possible because energy profiles exhibit recurring execution patterns driven by the underlying process, the product, and external conditions, and that these patterns can be systematically extracted and reproduced with data-driven techniques. Evaluation~2 shows that the most effective configuration segments a real reference execution into the characteristic steps of its shape, transfers the segments with DTW, and predicts their durations and levels with ML algorithms conditioned on external factors (\textit{ML Step DTW})—removing either the steps or the DTW alignment reduces its accuracy. By contrast, per-position regression over the aligned curves smooths the characteristic dynamics away, and sequence-to-sequence architectures yield consistently lower performance in our setting \citep{worrlein_using_2024}. The resulting profiles further provide a more realistic representation of industrial process behavior than methods that only use punctual median values per activity \citep{gonzalez_process_2025}, capturing the temporal interplay between process execution, product characteristics, and external factors that influence energy behavior.

\vspace{\baselineskip}
\textbf{Contribution to the literature on process-and-energy Digital Twins: causal, mechanistically grounded coupling.}

A third stream of literature builds Digital Twins that combine process models with energy information for manufacturing systems and their simulation \citep{khodadadi_data-driven_2024, khodadadi_automated_2026, gonzalez_process_2025}, but represents energy as a static, aggregated attribute of an activity (e.g., total or median consumption). We contribute the argument and the empirical evidence that this modeling has to be causal: an energy profile is generated by the execution of a specific machine activity, on a specific product, over a specific duration, and under specific external conditions. A model that disregards these causes may still reproduce aggregate statistics of the observed energy behavior, but it cannot represent the mechanism that gives rise to the profiles. Consequently, it cannot reliably simulate how those profiles change when the production plan, the product mix, or the external conditions change, and it is therefore unsuitable as a basis for what-if analysis.
 
Evaluation~3 supports this argument empirically: the best overall results are obtained by combining the process model with the energy model. Methods that disregard the process can still attain partly competitive aggregate metrics, but visual inspection reveals that the shape of the resulting profiles is severely distorted, with peaks and spikes that do not follow the real process structure. Process modeling is thus essential for energy profile generation, as it imposes a causally plausible and mechanistically grounded constraint on the resulting curves. As a result of stochastic variation in the simulation, the generated profiles are not identical to any specific historical trace; they are, however, plausible realizations for the facility under study—a guarantee that a Digital Twin ignoring the process cannot offer.
 
Beyond simulation accuracy, this coupling also improves \emph{understanding} of the underlying system. In industrial practice, process and energy are usually modeled separately, obscuring how process execution generates energy behavior. Modeling them jointly makes this link explicit: it reveals what a typical execution of an activity looks like, and which process characteristics, products, durations, and external conditions drive the resulting energy behavior. Our joint visualization of material flow, machine activities, and energy profiles supports both descriptive/diagnostic analytics—comparing real and simulated cases and inspecting prediction quality—and prescriptive uses, by illustrating how changes to a production schedule would affect the resulting energy behavior. This granular, causally explicit representation is what allows the Digital Twin to support informed decision-making, for instance identifying precisely which process, product, and external factors produced a given peak load at a given time.

\subsection{Practical Implications and Contribution}

For practitioners modeling industrial system operations, our findings argue against treating process and energy as separate modeling problems. The measures organizations use to improve industrial energy efficiency and meet net-zero targets \citep{iea_net_2023}—from waste heat recovery and electrification to peak load management, renewable integration, and energy-aware scheduling—all depend on detailed, reliable models of how energy demand is caused by process behavior. Existing Digital Twin approaches struggle to provide this: manual construction is costly and time-consuming \citep{sargent_verification_2010, nordgren_flexsim_2002, matloff_introduction_2008}, and where such twins are built automatically, energy is typically reduced to static, aggregated figures \citep{khodadadi_automated_2026} that obscure exactly the temporal and causal detail these measures require.
 
The Process and Energy Digital Twin offers a first blueprint for constructing coupled process-and-energy Digital Twins that address this need. As our evaluations show, process-aware modeling is the only family of approaches capable of simulating energy profiles that remain faithful to the underlying execution—aggregate accuracy alone is not sufficient once the goal is a twin organizations can act on. Because the twin is extracted automatically from event and sensor data organizations already record, it removes the manual effort that has limited the adoption of process-and-energy twins in practice. Because it represents energy as a continuous, causally grounded profile rather than an aggregated KPI, it captures how product mix, external conditions, and process execution jointly shape real-time energy demand—detail that measures such as peak load management or renewable integration cannot act on if it is averaged away. And because process and energy are coupled in simulation, the twin supports what-if analysis: organizations can evaluate the energy consequences of a planned schedule change or shift in product mix before implementing it, rather than only diagnosing energy behavior after the fact.
 
Our results also carry a practical message: the most accurate methods in our evaluations were not the most expensive ones. The proposed segment-based approach, built on standard ML algorithms, outperformed the far more training-intensive sequence-to-sequence models while requiring less training and prediction time (Appendix~\ref{app:computational_time}). The process model is cheaper still—discovery and duration fitting take seconds, and \textit{ml\_local} beats \textit{ml\_global} on both accuracy and simulation-time cost. Standard, cheap-to-train algorithms are therefore sufficient for a working Process and Energy Digital Twin, keeping the modeling effort within reach of typical industrial IT teams.
 
For practitioners, the framework is thus a blueprint for building realistic twins from data they already record, turning energy modeling from a manual, aggregated exercise into an automated, granular, and decision-ready one. This supports process- and energy-aware decision-making and, more broadly, organizations' efforts toward improved operational performance and sustainability.
 
\section{Conclusion, Limitations, and Future Work}
\label{Conclusion, Limitations, and Future Work}

In this paper we presented a framework for Process and Energy Digital Twins that jointly models discrete process execution and continuous, activity-level energy profiles from historical event and sensor data. Across three evaluations on simulated and real-world data from six industrial production processes, we showed that our model (1) can accurately reconstruct continuous energy profiles from discrete process activities, using a segmentation-and-alignment approach (\textit{ML Step DTW}) conditioned on external factors; (2) captures that energy profiles are strongly time-dependent on the process, as granular, time-aware process modeling—modeling machine activities explicitly and predicting activity and case durations with machine learning—substantially improves the fidelity of simulated case lead times; (3) achieves a more faithful representation of the industrial system by coupling process and energy modeling, rather than modeling either in isolation, which is necessary to obtain energy profiles that are both quantitatively accurate and qualitatively faithful to the shape of real industrial energy behavior; and (4) improves the understanding of the underlying system by jointly modeling and visualizing process and energy behavior, making explicit which process characteristics, products, durations, and external conditions drive a given energy profile.
 
This work focuses on the process and energy modeling of production processes, with particular emphasis on material flow, machine activities, and activity durations, and demonstrates how these factors influence energy profiles and must be incorporated to generate accurate energy predictions. It does, however, not consider all aspects that can be important for process modeling and simulation, such as the effect of work-in-process on the model or the waiting time before starting an activity \citep{camargo_learning_2023}, as these were not relevant for the industrial processes analyzed in this study. We focus on case-based modeling, leaving object-centric PM and its simulation for future work \citep{van_der_aalst_object-centric_2023, knopp_discovering_2023}. Extending the model with more advanced process-modeling techniques could enable finer-grained process and energy modeling.
 
We conducted a large-scale evaluation using both simulated and real energy profiles to empirically validate the methods; nevertheless, predicting the exact shape of highly volatile profiles remains challenging. Although our method outperforms the aggregated baselines, the simulated curves can still deviate from real-world anomalies. Future work should isolate individual energy profiles to determine which variables are necessary and sufficient for accurate shape prediction, identifying operational thresholds where data-driven modeling becomes unreliable, and, building on this, extend the current predictive focus toward what-if and counterfactual reasoning.
 
Finally, this paper focuses on the extraction and simulation of Process and Energy Digital Twins, but the resulting models were not evaluated against specific optimization benchmarks (e.g., net energy savings or financial ROI). Future research could deploy these modeling principles in prescriptive use cases, such as the optimal sizing of thermal storage tanks, battery integration, or energy-aware production scheduling.


\section{Data availability}
The code is available in an online repository. Data from the simulation is openly accessible in the repository. Data from the real processes is confidential and cannot be provided.

\section{Declaration of competing interest}
The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

\section{Declaration of generative AI and AI-assisted technologies in the manuscript preparation process.}

During the preparation of this work, the authors used Claude in order to paraphrase and support in software development. After using this tool, the authors reviewed and edited the content as needed and take full responsibility for the content of the published article.

\section{Acknowledgments}



\appendix
\onecolumn
\section{Simulation of the synthetic process (process\_1)}
\label{app:process1_sim}

The synthetic dataset \textit{process\_1} is generated by a discrete-event simulation of a hospital infusion-bag production line, coupled with physical models for the thermal equipment. The complete implementation of the generation of this dataset is in the online repository for complete reproducibility. Since both the process logic and the energy physics are defined by construction, the dataset provides a known ground truth for the link between activities and energy behavior. The generator produces the same three artefacts the framework consumes for the real processes: an event log, a minute-resolution energy time series, and a production plan with one row per batch.

The line consists of six stations executed in sequence: water supply, distillation, bottling, autoclaving, individual packaging and warehousing. Every station is a resource with its own availability, and a batch waits until the station is free. Within each station, the machine executes its internal activities, whose durations are sampled from normal distributions around activity-specific base durations; for the stations that handle the product, the duration additionally scales with the batch volume, so that volume drives the case duration. Table~\ref{table:process1_simulation} summarizes the stations, their resources and machine activities, the base durations, and the built-in control-flow variability.

\begin{table}[width=.9\linewidth,cols=5,pos=h]
\caption{Overview of the discrete-event simulation of \textit{process\_1}. Base durations refer to the standard recipe and are the means of the sampled normal distributions; the autoclave phase durations follow from its sampled cycle length and fixed phase split.}
\label{table:process1_simulation}
\footnotesize
\begin{tabular}{lclll}
\toprule
station & \makecell{parallel\\resources} & machine activities & \makecell[l]{base durations\\(min)} & \makecell[l]{volume-\\scaled} \\
\midrule
Water supply         & 1 & prepare, working                & 0.06, 1.2        & \checkmark \\
Distillation         & 1 & \makecell[l]{preparation, production,\\emptying, maintenance$^{\ast}$} & 3, 25, 4, 15 & \checkmark \\
Bottling             & 1 & prepare, working, cool          & 0.06, 0.67, 0.06 & \checkmark \\
Autoclaving          & 3 & prepare, heat, hold, cool       & \makecell[l]{2; cycle $\approx$ 60,\\split 27/53/19\,\%} & \checkmark \\
Individual packaging & 1 & prepare, working                & 5, 5             & \checkmark \\
Warehousing          & 1 & prepare, working                & 5, 5             & -- \\
\bottomrule
\end{tabular}

\vspace{0.5em}
\parbox{\linewidth}{%
\footnotesize
300 batches are released every 4 hours, each with a batch volume drawn uniformly from 200--800~L that scales the durations of the volume-scaled stations. Control-flow variability: 20\% of batches skip Individual packaging; 40\% fail the quality check at Warehousing and rework packaging and warehousing; 15\% of autoclave cycles trigger a rework of heat--hold--cool (half of them including prepare); $^{\ast}$maintenance is inserted after emptying with 1\% probability. Batches queue for the earliest free resource; the autoclave is picked by a weighted random choice favouring autoclave~2.
}
\end{table}

\paragraph{Physical generation of the energy curves}
The energy time series contains six sensors: steam and cooling demand of the distillation, steam and cooling-water demand of the autoclaves, and the electrical power of bottling and individual packaging. Each activity execution generates its curve from a physical model of the equipment. The autoclave follows a thermodynamic model of the sterilization cycle: during \textit{heat}, a constant steam mass flow produces a power plateau $\dot{Q} = \dot{m}_{\mathrm{steam}} \, h_{\mathrm{vap}}$ (latent heat from steam properties) that tapers off as the vessel, modelled with its heat capacity and heat-loss coefficient, approaches the sterilization temperature; during \textit{hold}, the steam only compensates the heat loss $\dot{Q} = A U \,(T_{\mathrm{steri}} - T_{\mathrm{amb}})$, with small valve noise; during \textit{cool}, the cooling-water power $\dot{Q} = \dot{m}_{\mathrm{cool}} \, c_p \,(T_{\mathrm{out}} - T_{\mathrm{in}})$ follows a smooth rise-and-fall of the outlet temperature whose integral equals the energy required to cool the charge. The distillation demands come from a steady-state energy balance (feed heating plus vaporization of the light fraction for the steam side; condensation and product cooling for the cooling side), held constant during \textit{production}, ramped during \textit{preparation} and \textit{emptying}, with $\pm5\%$ noise. Bottling and individual packaging are constant electrical base loads with Gaussian noise. The steam and cooling-water flows scale with the batch volume, so the plateau and peak levels carry the volume of each batch, and between executions every sensor keeps a small noisy standby draw instead of falling to zero. For this process, the physics is solved at a fixed ambient temperature, so the recorded weather columns do not drive the energy demand. 


\section{Profile-generator baseline}
\label{app:schedule_profile_baselines}

The \textit{Profile-generator} (Table~\ref{tab:energy_profile}) predicts a complete case profile without running the Petri-net simulation. It is fitted per sensor and per process, on the training cases only, at case-level granularity: one real energy curve per case, resampled by linear interpolation onto a fixed number of canonical positions along the case's fractional progress (0 to 1).

The generator is population-level only and uses no case attributes. A Normal distribution (mean, std) is fit per canonical position across all training cases' resampled curves, following the ``DES with stochastic distributions'' baselines in the energy-DES literature \citep{kouki_input_2017}. Prediction: one independent sample drawn from these per-position Normal distributions, regardless of the case.

The time axis is each test case's own predicted total duration, from the same duration-prediction pipeline used for the duration-corrected process-model simulations, falling back to the population median duration when no prediction is available. The curve values of the \textit{Profile-generator} are therefore independent of the case and only stretched to its predicted lead time.



\section{Computational time analysis}
\label{app:computational_time}


All experiments were run on a Linux workstation (Ubuntu 20.04.6 LTS) with a 16-core Intel Core i7-9800X CPU at 3.80\,GHz, 125.5\,GB of RAM, and two NVIDIA TITAN RTX GPUs with 27.77\,GB of memory each (NVIDIA driver 535.171.04, CUDA 12.2).

Table~\ref{tab:timing_process_stages} shows that the process-model stages are cheap throughout. Within process discovery, the Heuristic and Inductive miners dominate at 4.86 and 4.77 s per process, roughly three times the 1.73 s of the Alpha miner. For the activity-duration models, training ml\_local costs 4.26 s per process against 0.91 s for ml\_global, almost five times more, since it fits the ML algorithm once per activity instead of once on all events of the process. At simulation time the relationship reverses: ml\_global takes 5.25 ms per case while ml\_local takes 2.28 ms, so the per-activity variant is more expensive to train but cheaper to simulate with. All simulation variants remain in the low milliseconds per case.

Table~\ref{tab:timing_results} shows that the cost of the ML-based curve methods lies in training, not in inference. The proposed ML Step DTW method trains its 390 curve models in 247.35 CPU minutes, about 38 CPU seconds per model, less than a third of the cost of the ML variants without step segmentation, while every method generates a curve in tens of milliseconds at most.


\vspace{1em}

\noindent
\begin{minipage}{\textwidth}
\centering

\captionsetup{
    justification=centering,
    singlelinecheck=false,
    format=plain
}

\captionof{table}{Computational cost of the process-model stages.}
\label{tab:timing_process_stages}

\vspace{-0.5em}

\small
\begin{tabularx}{\linewidth}{>{\raggedright\arraybackslash}X|r|r}
\toprule
Stage & Cost & Total (s) \\
\midrule
Process discovery -- Alpha miner & 1.73 s / process & 10.4 \\
Process discovery -- Heuristic miner & 4.86 s / process & 29.2 \\
Process discovery -- Inductive miner & 4.77 s / process & 28.6 \\
Case-duration model (budgeting) & 0.50 s / process & 3.0 \\
Duration ML algorithm training -- ml\_global (one per process) & 0.91 s / process & 5.4 \\
Duration ML algorithm training -- ml\_local (one per activity) & 4.26 s / process & 25.5 \\
Simulation -- sampled durations & 1.06 ms / case & 5.9 \\
Simulation -- ML global durations & 5.25 ms / case & 29.1 \\
Simulation -- ML per-activity durations & 2.28 ms / case & 12.6 \\
\bottomrule
\end{tabularx}

\vspace{0.5em}

\parbox{\linewidth}{%
\footnotesize
Runtime of the process-model stages, same run. Fitting stages are per process (6 processes), simulation per generated case (test split). Cost is single-process wall clock, a ratio rather than an absolute deployment figure; Total is what this run spent on the stage, so it scales with the experiment grid (every net crossed with every process). The simulation rows are split by the activity-duration variant of the process model: durations sampled from the training distribution, ml\_global, or ml\_local. The two duration-ML rows are training cost: they share one training pass and its feature preparation; ml\_global fits the ML algorithm once on all events of the process, ml\_local once per activity, so its cost scales with the number of activities.
}

\end{minipage}

\vspace{1em}

\vspace{1em}

\noindent
\begin{minipage}{\textwidth}
\centering

\captionsetup{
    justification=centering,
    singlelinecheck=false,
    format=plain
}

\captionof{table}{Computational cost of the curve-generation methods.}
\label{tab:timing_results}

\vspace{-0.5em}

\small
\begin{tabularx}{\linewidth}{>{\raggedright\arraybackslash}X|c|c|c|c}
\toprule
Method & \makecell{Curve\\models} & \makecell{Training\\total (CPU min)} & \makecell{Training per\\model (CPU s)} & \makecell{Inference per\\curve (ms)} \\
\midrule
ML Step DTW (proposed) & 390 & 247.35 & 38.05 & 22.6 \\
Median per Activity \& Sensor & 390 & \textbf{$<$0.01} & \textbf{$<$0.01} & \textbf{6.6} \\
ML DTW (no steps) & 390 & 869.50 & 133.77 & 21.2 \\
ML (no DTW) & 390 & 865.86 & 133.21 & 16.9 \\
Seq2Seq DTW (aligned) & 390 & 164.05 & 25.24 & 40.1 \\
Seq2Seq (DTW-scored) & 390 & 678.28 & 104.35 & 36.0 \\
\bottomrule
\end{tabularx}

\vspace{0.5em}

\parbox{\linewidth}{%
\footnotesize
Cost of each curve-generation method on the run of Table~\ref{tab:individual_profile_realism}. A curve model is one predictor per (sensor, activity, object). Training is summed over the 16 workers of the training pool, inference is wall clock over the test curves of a single process. Ratios, not absolute deployment figures. \textbf{Bold} = cheapest per column.
}

\end{minipage}


\clearpage
\onecolumn
\section{Individual profiles visualization}
\label{app:Individual_profiles}

\begin{center}
\includegraphics[width=\textwidth]{10_individual_profiles.pdf}
\captionof{figure}{Individual energy profiles on the test set: each panel shows one activity execution on one sensor, with the real curve (black) and the predictions of all methods. For confidentiality, sensor and activity names are anonymized and the curves are normalized to 0--1, which also allows comparing shapes across sensors of different scales. The corner value is the realism error of \textit{ML Step DTW} on that curve (0 = perfect, lower is better).}
\label{fig:individual_profiles}
\end{center}

\twocolumn


\bibliographystyle{cas-model2-names}

\bibliography{references}




\end{document}

