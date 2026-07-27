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
\usepackage{array}
\usepackage{caption}




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
\shorttitle{PM and energy}
\shortauthors{}

\title [mode = title]{This is a specimen $a_b$ title}                      
\tnotemark[1,2]

%\address[1]{, Street 129, 1043 NX Amsterdam, The Netherlands}


\cortext[cor1]{Corresponding author}
\cortext[cor2]{Principal corresponding author}
\fntext[fn1]{}
\fntext[fn2]{}

\nonumnote{}

\begin{abstract}
This template helps you to create a properly formatted \LaTeX\ manuscript.

\noindent\texttt{\textbackslash begin{abstract}} \dots 
\texttt{\textbackslash end{abstract}} and
\verb+\begin{keyword}+ \verb+...+ \verb+\end{keyword}+ 
which
contain the abstract and keywords respectively. 

\noindent Each keyword shall be separated by a \verb+\sep+ command.
\end{abstract}

\begin{graphicalabstract}
%\includegraphics{figs/cas-grabs.pdf}
\end{graphicalabstract}

\begin{highlights}
\item Research highlights item 1
\item Research highlights item 2
\item Research highlights item 3
\end{highlights}

\begin{keywords}
quadrupole exciton \sep polariton \sep \WGM \sep \BEC
\end{keywords}

\maketitle

\section{Introduction}
\label{sec:introduction}

% -----Argumentationslinie

% - Industrial energy consumtion is high we need to make it more efficient/reduce it...
% - Aprpoaches to improve enrgy demands requrie energy demand modelling...
% - Digital models have been used extensively to modell proccesses, even with energy info, 
% - However, approaches focus on the material flow modelling and energy, if it appears überhaupt, is an "attrribute" (Punctual energy demands): not accurate process with enery profile modelling (the time series)...
% - We come in with process and energy profile modelling and simulaiton...

%#1 Domain and importance
Industrial processes account for approximately 40\% of global end-use energy consumption \citep{iea_world_2024}, with emissions primarily stemming from electricity use and industrial process heat. In Europe, for instance, electricity meets about 34\% of industrial final energy demand, while the remaining 66\% is dedicated to process heat \citep{de_boer_strengthening_2020}. Driven by international climate targets and net-zero commitments \citep{iea_net_2023}, industries must accelerate the improvement of energy efficiency and the reduction of energy-related emissions.

%#2 Overall problem/situation
Various approaches address these goals, including waste heat recovery \citep{kemp_pinch_2007, klemes_handbook_2022}, electrification of heat generation \citep{ashabi_assessing_2025, knorr_electrification_2025}, electricity peak load management, reduction of power demand, integration of fluctuating renewable energy sources \citep{pee_decarbonization_2018, wenzel_energy-related_2024}, and energy-aware production scheduling \citep{shao_systematic_2024}. However, the effective implementation and optimization of these approaches critically depend on detailed and reliable energy demand models of industrial processes, in order to identify optimization potentials and evaluate their impact. This, in turn, requires a clear understanding of how energy patterns are caused by specific process behavior.

%#3 Conclusions of existing literature
A key enabler to address this need is the concept of digital twins: software-based representations of physical systems that enable advanced analysis, process improvement, and data exchange between the physical object and the digital copy \citep{liu_digital_2024}. Within Industry 4.0 and 5.0, digital twins have become central to modeling and improving industrial processes, with particular emphasis on sustainability and energy efficiency \citep{soori_internet_2023, xu_industry_2021, tao_data-driven_2018}. As manually modeling the components of such twins is time-consuming and difficult given the complexity of production processes \citep{ssargent_verification_2010, nordgren_flexsim_2002, matloff_introduction_2008}, prior research has increasingly leveraged data-driven approaches to automatically extract and simulate process models from historical data, more recently also incorporating energy-related aspects into these models \citep{khodadadi_data-driven_2024, khodadadi_automated_2026}.

%#3 Problem with the current literature
However, existing approaches usually represent energy in a simplified manner, e.g., as discrete or aggregated energy consumed during a specific production event \citep{??}. Such representations neglect the underlying energy behavior, i.e., the continuous time-series energy profiles and the dynamic effects of the process events on them \citep{wenzel_energy-related_2024}. Consequently, the temporal effects of process execution or external factors like weather on real-time energy behaviour remain uncaptured. This limits the ability of digital twins to accurately reflect real energy behavior, oversimplifying the production system and constraining their potential to identify and realize meaningful energy efficiency improvements \citep{lal_nathan_s_accounting_2018, gonzalez_process_2025}.

%#5 How this study addresses the gap
To address this gap, we propose a data-driven process and energy modeling framework for industrial systems that integrates Process Mining methos \citep{van_der_aalst_data_2016} with causally grounded, continuous energy profiles trained with Machine Learning methods \citep{hastie_introduction_2009}. Together, these components form the foundation for a facility's Process and Energy Digital Twin. By explicitly linking discrete process execution with continuous energy behavior from historical event logs, the framework captures complex real-world dynamics—including interactions among process events, product attributes, external factors, and energy usage. Ultimately, the Digital Twin enable better understanding of the manufacturing system and its accurate simulation.

%This paper addresses this gap by proposing a data-driven process and energy modeling framework for industrial systems that jointly captures discrete process events and continuous energy profiles. By explicitly linking process behavior to its resulting energy consumption, the framework aims to reflect the complex real-world characteristics of industrial processes, including how process events, product characteristics, external factors, and energy consumption interact.
%The approach enables the simultaneous modeling of the discrete process behavior and how it causes the underlying dynamic energy behavior. 

%#6 Study setting, data, and methods.
Our methodology first extracts process models using process mining, capturing material flow as well as machine activities and their durations. These process models provide the basis for accurate energy modeling: building on them, we extract continuous energy profiles through data transformations and Machine Learning methods. Combining both results in a joint process and energy digital twin, which we validate using one simulated and three real-world industrial processes.

%main findings
The results show that the process and energy digital twin can reproduce new process instances and generate accurate energy profiles. By grounding these profiles directly in their causal sources–process execution, duration, product characteristics, and external factors such as weather–the framework captures a more faithful representation of real energy behavior than approaches relying on discrete or aggregated energy values.

%contribution
These findings extend existing knowledge by demonstrating how data-driven process and energy modeling can lead to a more faithful representation of real-world production systems, thereby expanding the literature on industrial process and energy modeling. The proposed framework effectively bridges and contributes to several literature streams, namely process mining, Machine Learning-based energy modeling, and simulation methods. From a theoretical perspective, the integrated modeling of discrete process behavior and continuous energy profiles provides a novel foundation for analyzing how production schedules and external factors affect process execution and its resulting energy behavior. From a practical perspective, the framework strengthens the capability of digital twins to support data-driven process and energy management and better-informed decision-making in industry.

% #9 Paper outline
Section~\ref{sec:related-work} introduces important background concepts and reviews the related work. Section~\ref{sec:methodology} presents the proposed process and energy digital twin framework, describing its extraction from real-world data and its application for simulation. Section~\ref{sec:evaluation} describes the evaluation procedure. Section~\ref{sec:results} reports the case study and evaluation results, followed by the discussion (Section~\ref{Discussion}) and conclusions (Section~\ref{Conclusions}).

\section{Background and Related Work}
\label{sec:related-work}
\subsection{Digital twin and simulation}

%A \textit{system} is defined as a collection and interaction of components to achieve a purpose, for instance, a manufacturing plant. 

To investigate a system such as a manufacturing plant, its structural and behavioral assumptions must first be formalized into a \textit{model}---expressed through mathematical formulas, logical relationships, or software-based \textit{digital models} \citep{wainer_discrete-event_2017}. Through \textit{simulation}, a computer then dynamically executes such a digital model to imitate the system's real-world operation over time, generating data that can be used to analyse and optimise its performance \cite{law_simulation_2015}.

If an automated, one-way data flow is established from the physical system to the digital model, it acts as a digital shadow; and if two-way data integration, it is considered to be a \textit{digital twin} \citep{grieves_digital_2014, liu_digital_2024}.

There are several types of system's modeling and simulation paradigms that can be used for digital twins. Each is suited to different kinds of problems, goals, and system types. Dynamic simulations are for systems that evolve continuously over time, often using mathematical models, and is commonly applied in physical and engineering contexts \citep{cellier_continuous_2013}. Discrete-event simulation (DES) is widely used for systems in which events produce system changes at distinct points in time \citep{wainer_discrete-event_2017}. Agent-based simulation models the behavior and interactions of autonomous agents, making it suitable for complex adaptive systems, like social interactions or biology \citep{bonabeau_agent-based_2002}. System dynamics, a subset of continuous simulation, emphasizes feedback loops and accumulations in complex systems, particularly in policy and organizational studies \citep{bala_system_2017}. There are also hybrid simulation approaches that use a combination of some of the previous ones.

This paper examines both the discrete process dynamics and the continuous energy behaviour of manufacturing systems. To capture these dynamics within a digital twin for subsequent simulation, we integrate data-driven methods with established discrete and continuous modelling techniques. The subsequent sections detail these paradigms and explore their respective data-driven modelling approaches.

\subsection{Discrete event simulations and data-driven process modelling}

\citep{wainer_discrete-event_2017} \citep{wainer_discrete-event_2018}?

Discrete event simulation (DES) is a method used to capture system dynamics by advancing the state only at discrete points in time, driven by events such as arrivals, service completions, or failures. Instead of evolving continuously, the system transitions between states when events occur, and these events are typically managed through an event-scheduling mechanism (e.g., a future event list) \citep{wainer_discrete-event_2018}. DES models usually consist of entities (e.g., jobs or customers), resources (e.g., machines or servers), queues, and routing logic that governs how entities move through the system. This paradigm is widely used to represent systems such as manufacturing processes, logistics networks, and service operations \citep{wainer_discrete-event_2017}.

For a DES, it is necessary to create a model of the system by defining key components, such as entities, resources, and queues—alongside the logic governing their interactions. Modelers must specify potential events and implement a scheduling mechanism, typically based on a future event list, to determine the sequence and timing of state changes \citep{wainer_discrete-event_2018,brailsford_discrete-event_2014}. Although these models are typically built using dedicated software \citep{nordgren_flexsim_2002, matloff_introduction_2008}, the development process remains challenging and time-consuming. This difficulty arises from the need to explicitly specify stochastic behavior, routing rules, and resource constraints, which often results in complex models \citep{sargent_verification_2010}. As the system´s complexity grows, models become harder to build, validate, and maintain. These issues can limit scalability and hinder the practical adoption of DES in complex real-world systems.

To reduce this effort and better reflect real-world production, data-driven approaches, particularly \textit{Process Mining (PM)} have gained increasing attention. PM techniques, especially process discovery algorithms, can automatically derive process models from historical real produciton data. These methods capture both the model structure and its parameters, including control-flow patterns, resource behavior, and performance characteristics \citep{dumas_fundamentals_2018,weske_business_2019, dreher_application_2021, van_der_aalst_process_2022, rozinat_discovering_2009, camargo_automated_2020, castiglione_automated_2024}. In smart manufacturing production systems, this level of automation is commonly regarded as a prerequisite for operational digital twins capable of keeping pace with frequent changes in production routings and plans \citep{friederich_framework_2022,uhlemann_digital_2017,zheng_application_2019}.

The data source for PM is the \textit{event log}, which captures the history of the process in a structured format. Formally, an event log $L$ consists of a set of recorded events that represent the actual step-by-step execution of a system. Let $E$ be the universe of all events. Every event $e \in E$ is characterized by a specific timestamp $\pi_{time}(e)$, and refers to a particular case $c \in C$ (i.e., a process instance) via the mapping $\pi_{case}(e) = c$, as well as a specific activity $a \in A$ via $\pi_{act}(e) = a$. All events belonging to the same case form a time-ordered sequence known as a trace $\sigma = \langle e_1, e_2, \dots, e_n \rangle$, such that $\pi_{time}(e_i) \leq \pi_{time}(e_{i+1})$ for all $1 \leq i < n$. Furthermore, events can carry extended information as attributes. Common extensions of event logs include the resource executing the activity $\pi_{res}(e)$ (e.g., an object or person), a lifecycle transition $\pi_{life}(e)$ (i.e., information about state changes such as \textit{start} or \textit{complete}), or domain-specific recorded data elements $\pi_{data}(e)$ (e.g., the size of an order) \citep{daniel_process_2012, van_der_aalst_process_2016, van_der_aalst_process_2022}.


%%%%%However, applying PM to data from manufacturing systems remains challenging because of the complex interactions among multiple entities, such as machines states and material flow, the loss of uniqueness of process instances caused by the transformation of the product in the product, and the unstructured, cascading, and non-linear nature of manufacturing processes \citep{dreher_application_2021, lugaresi_automated_2023}. As a consequence, models extracted with traditional event-based process mining often become spaghetti-like \citep{van_der_aalst_process_2011} and fail to capture the manufacturing system at an appropriate level of abstraction \citep{lugaresi_automated_2021}.

%%%%To address these challenges, previous work has proposed different modelling strategies. Some studies separate system modelling into the material flow and the states occurring inside machines, particularly for machine reliability analysis \citep{friederich_data-driven_2022, friederich_framework_2022, friederich_process_2022}. In these approaches, the material flow is typically analyzed using the product as case identifier, while machine states are modeled separately and do not necessarily require a case identifier. Other studies focus the modelling of the manufacturing system on the flow of material and on key production events, such as when a product arrives at a machine, starts processing, and leaves the machines \citep{castiglione_automated_2024}. 

PM extracts process models using discovery algorithms such as Alpha Miner \citep{van_der_aalst_workflow_2004}, which is one of the earliest approaches and identifies basic sequential, parallel, and causal relations between activities, although it is sensitive to noise and therefore less suitable for complex real-life logs; Heuristic Miner \citep{weijters_process_2006, weijters_flexible_2011}, which improves robustness by relying on frequency-based relations and is therefore better suited to noisy event data; Inductive Miner \citep{leemans_discovering_2013, leemans_scalable_2015}, one of the most commonly used algorithms because it generates structured and sound process models that are relatively easy to interpret.

Depending on the method and implementation, process discovery algorithms can produce various process model representations, including heuristic nets, process trees, directed-follow graphs (DFGs), Business Process Model and Notation (BPMN), and Petri nets \citep{van_der_aalst_data_2016}. Among these, Petri nets are the most common representation for models that can be directly utilized for simulation in manufacturing systems \citep{castiglione_automated_2024, bause_stochastic_2002, simon_adapting_2018}. 

In general, Petri-Net-based models describe several core components: the \textit{process control flow}, which captures causal dependencies, sequential execution, and parallel interactions among activities; \textit{activity durations}, representing the time required to complete individual tasks; \textit{transitions}, which correspond to the observed activities recorded in the event log; and \textit{places}, which represent the conditions or system states that enable and synchronize these activities \citep{bause_stochastic_2002}.

There are also extensions of standard Petri nets tailored to specific system complexities. For example, Stochastic Petri nets (SPNs) account for the inherent stochasticity of manufacturing processes—such as probabilistic transitions and control-flow variability—by associating exponentially distributed firing delays with transitions \citep{bause_stochastic_2002, simon_adapting_2018, khodadadi_automated_2026}.  Stochastic Timed Petri nets (STPNs) \citep{wang_timed_2012} generalize this by allowing arbitrary probability distributions for firing delays, enabling activity durations to be modeled as random variables following empirically derived or domain-specific distributions. In this work, we use as basis STPNs \citep{wang_timed_2012}: 

A STPN is defined as a tuple $\mathcal{N} = (N, M_0, M_f, \mathcal{G}, \pi)$, where the underlying Petri net structure is given by $N = (P, T, F, W)$. Specifically, $P$ and $T$ are disjoint finite sets of places and transitions, $F \subseteq (P \times T) \cup (T \times P)$ is the flow relation, and $W : (P \times T) \cup (T \times P) \rightarrow \mathbb{N}$ assigns arc weights, where $W(x,y) > 0$ if and only if $(x,y) \in F$, so that non-existing arcs have weight $W(x,y) = 0$. The functions $M_0, M_f : P \rightarrow \mathbb{N}$ denote the initial and final markings, respectively. To model temporal and stochastic behavior, $\mathcal{G} = \{D_t\}_{t \in T}$ defines the set of firing-time distributions associated with the transitions, and $\pi : T \rightarrow [0,1]$ defines routing probabilities to resolve conflicts, satisfying $\sum_{t \in \mathcal{C}} \pi(t) = 1$ for every set $\mathcal{C} \subseteq T$ of mutually conflicting transitions. A transition $t \in T$ is \emph{enabled} at marking $M$ if $M(p) \geq W(p,t)$ for all $p \in \bullet t$ (where $\bullet t$ denotes the preset of $t$). If multiple enabled transitions compete for the same tokens, conflicts are resolved by sampling from $\pi$. In this work, the routing probabilities are induced by non-negative transition weights $w : T \rightarrow \mathbb{R}_{>0}$, normalised over the conflicting transitions, $\pi(t) = w(t) / \sum_{t' \in \mathcal{C}} w(t')$. When an enabled transition $t$ fires, the marking updates to $M'(p) = M(p) - W(p,t) + W(t,p)$ for all $p \in P$. The execution process terminates once the marking $M$ reaches the final marking $M_f$.


The \textit{stochastic} part is captured by $\pi$, which assigns routing  probabilities to transitions at conflict points (XOR-splits). Rather than deterministically following a fixed path, the net samples which transition fires next based on these probabilities, reflecting the inherent control-flow variability of real processes.

The \textit{timed} part is captured by $\mathcal{G} = \{D_t\}_{t \in T}$, which associates each transition with a probability distribution over firing delays. When a transition fires, its duration is sampled from $D_t$, allowing activity durations to follow arbitrary distributions fitted to observed data.

\subsection{Dynamic simualtions and data-driven energy modelling} 

Dynamic simulation is an umbrella term that encompasses models of systems whichs state evolves continuosly over time. Within this broad category, mathematical models can be classified according to how time is represented. In continuous simulation, system state variables evolve smoothly over time and are governed by differential equations \citep{cellier_continuous_2013}. These models capture the dynamic behaviour of physical systems as a continuous temporal process and are therefore well suited for domains in which variables change without interruption, such as mechanical, chemical, biological, and energy systems. Typical applications include fluid dynamics, thermal processes, and mechanical motion. Continuous modelling is also widely used in energy systems, where quantities such as power, temperature, and pressure evolve continuously over time \citep{grigsby_power_2007, bergman_fundamentals_2011}.

Another class of mathematical models are the discrete-time mathematical models represent time as a sequence of discrete steps, with system evolution described through difference equations. They are widely used in engineering systems, particularly in digital control, where computations are performed at fixed sampling intervals and require finite time to determine the next state \citep{brunton_data-driven_2022}. Because of this, it is natural to model time as equally spaced steps. These models can either be inherently discrete or obtained by discretizing continuous-time model. In practice, even continuous models are often converted into discrete form for numerical solution, since analytical solutions are frequently not available, and numerical integration methods approximate continuous system behavior by evaluating it at discrete time steps, avoiding the need to represent infinitely many state changes within a finite time interval \citep{cellier_continuous_2013}.

Constructing continuous models of energy systems requires deriving governing equations from first principles, such as conservation laws (mass, energy, momentum) and thermodynamic relations \citep{brunton_data-driven_2022, bergman_fundamentals_2011}. In practice, this approach demands deep domain expertise and a thorough understanding of the underlying physical mechanisms. Although, these models are typically developed using dedicated dynamic simulation software \citep{klee_simulation_2018}, this task can be labor-intensive. Furthermore, it introduces modelling errors through unavoidable simplifying assumptions, particularly when representing complex systems with poorly understood physical dynamics \citep{ghadami_data-driven_2022}.

These limitations have steered interest into data-driven or \textit{Machine Learning (ML)} modelling techniques, which extracts a system´s model directly from measured data \citep{brunton_data-driven_2022}. For instance, regression and time-series algorithms can effectively approximate complex, nonlinear energy system dynamics without explicit knowledge of the underlying physical equations \citep{bishnu_computational_2023, van_den_hof_system_2020}. 

The use of ML techniques has been explored to predict the total energy consumption of production facilities \cite{mosavi_energy_2019}, as well as energy consumption at the machine level \cite{he_generic_2020, zhang_data-driven_2021, mawson_deep_2020}. 

In general, these approaches can be described as supervised learning problems in which a target variable \(y\) is modelled as a function of a set of input variables \(\mathbf{x}\). Depending on whether the model predicts a single value or a time-varying output, this takes two forms:
\begin{equation}
y = f(\mathbf{x}) + \varepsilon, \qquad \text{or} \qquad y_t = f(\mathbf{x}_t) + \varepsilon_t,
\end{equation}
where in the first case $y$ is a scalar target predicted from a static input vector $\mathbf{x}$, and in the second case $y_t$ and $\mathbf{x}_t$ denote the output and input at the same time step $t$, respectively. 

It is important to note that in the literature, the execution of dynamic models in simulations is generally referred to as "simulation," whereas the application of ML models for the same task is usually termed "prediction." This distintions also persists even when the dynamic models have no stocastic componet or when the ML predictions serve a simulative purpose, such as playing an scentario of input parameters and observe the results. To maintain consistency with established terminology, this paper uses the term "prediction" to describe the outputs of ML models for energy profiles, though in this context, the concepts function interchangeably with dynamic simulation.

%This drastically minimizes modelling overhead, enabling simulation even when the physical mechanisms are poorly understood or too intricate to formalize in a mathematical model \citep{ghadami_data-driven_2022, wang_hybrid_2022}.T

%hese methods generally produce discrete predictive models—for example, mapping observed states up to time $t$ to a predicted output at step $t+1$. 

%In the case of production facility energy forecasting, the input vector may include production schedules, climatic conditions, thermal properties of the building, and building behaviour and use, while the target variable may correspond to energy consumption, temperature, or humidity \cite{he_generic_2020, zhang_data-driven_2021, mawson_deep_2020}. 

%For example, \cite{he_generic_2020} used ML methods to predict the power consumption of milling and grinding machines based on production parameters such as spindle speed. They found that their methods can allow companies an effective energy management. Similarly, \cite{zhang_data-driven_2021} applied ML methods for industrial robots to model the relationship between operating parameters and energy consumption, and then used a genetic algorithm to optimize these parameters for energy savings. They found that this approach achives substancial energy savins in the robot´s operation. Also, \cite{mawson_deep_2020} predicted the energy consumption of a simulated production environment together with temperature and humidity, forecasting energy consumption at one-hour intervals, they found that this can eb done with a high accuracy and highlighting the importance of environmental conditions in the energy behaviour of the facility.

\subsection{Related Work}

Our work bridges the fields of discrete-event and dynamic simulation, specifically focusing on data-driven modelling techniques such as Process Mining (PM) and Machine Learning (ML). To contextualize our contribution, Table~\ref{tab:literature_comparison} summarizes related literature across five dimensions: the use of process simulation, the application of data-driven process modelling, general energy considerations (for instace as total value at the end of a simualtion), the ability to simulate energy profiles, and the use of data-driven energy profile modelling.

Studies have worked on the automated process model extraction generates process models from historical event logs with PM. For instance, \cite{friederich_framework_2022} applied PM techniques to model material flows and machine reliability, demonstrating that these models enable data-driven digital twins for manufacturing system simulations. Similarly, \cite{lugaresi_automated_2023} proposed a PM algorithm to construct digital twins for systems with complex material flows, validating its effectiveness in a real-world manufacturing environment. Further expanding on this, \cite{castiglione_automated_2024} introduced an event-centric process mining framework that tracks material entering and leaving machines, enabling the rapid, automated generation of digital models while remaining robust under low-data conditions.

Other studies have used DES to include energy consumption data. For example, \cite{kohl_discrete_2014} expanded DES with energy models to generate full energy profiles rather than fixed values, showing that this improves energy predictions for production lines and full factories. Similarly, \cite{kouki_input_2017} reviewed the literature on incorporating load profiles into discrete event simulations and proposed using stochastic distributions. They showed that their approach results in only a small deviation from actual energy measurements.

Furthermore, a few studies have combined discrete event data or simulations with machine learning to predict continuous energy profiles. For instance, \cite{woerrlein_method_2020} applied sequence-to-sequence machine learning models to predict time-series energy consumption directly from numerical control (NC) code. Building on this, \cite{worrlein_using_2024} showed that sequence-to-sequence models can capture energy curve patterns effectively, which significantly improves prediction accuracy.

Finally, other studies use process mining to extract process models directly from event data and add specific energy-related factors, such as power consumption, waste generation, and CO2 emissions. For instance, \cite{hodadadi_data-driven_2024} developed energy-oriented digital twins to better understand energy behavior in smart factories, showing how operational schedules interact with energy use. Building on this, \cite{khodadadi_automated_2026} proposed a framework to automatically extract Petri nets with multidimensional properties like time, energy consumption, and waste generation. In their experiment, they showed that their approach can simulate what-if scenarios and reduce energy consumption without affecting production output. Similarly, \cite{gonzalez_process_2025} extracted process models from manufacturing event logs and combined them with specific energy demands. Combining event logs with energy data allowed them to generate step-wise energy profiles, improving energy analysis and making it easier to evaluate heat recovery potential.


\begin{table*}[H]
\centering
\caption{Comparison of the proposed approach with related literature}
\label{tab:literature_comparison}
\resizebox{\textwidth}{!}{%
\begin{tabular}{@{}p{4.5cm}p{7cm}ccccc@{}}
\toprule
\textbf{Study} & \textbf{Description} & \textbf{Process simulation} & \textbf{Data-driven process modelling} & \textbf{Energy consideration} & \textbf{Energy profiles prediction} & \textbf{Data-driven energy profile modelling} \\ \midrule

\cite{friederich_framework_2022} & Automatic process digital twin generation with PM. & \checkmark & -- & \checkmark & \checkmark & -- \\ \addlinespace
\cite{kohl_discrete_2014} & DES extending material flows with energy consumption information. & \checkmark & -- & \checkmark & \checkmark & -- \\ \addlinespace
%Zavanella et al. (2015) \citep{zavanella_energy_2015} & Uses queuing theory to estimate power peak probabilities. & -- & -- & \checkmark & -- & -- \\ \addlinespace
\cite{kouki_input_2017} & DES modeling energy via stochastic distributions. & \checkmark & -- & \checkmark & \checkmark & -- \\ \addlinespace
\cite{woerrlein_method_2020} & DES triggers a predictive model curve. & \checkmark & -- & \checkmark & \checkmark & \checkmark \\ \addlinespace
\cite{camargo_discovering_2021} & Automated generative DES discovery from event logs. & \checkmark & \checkmark & -- & -- & -- \\ \addlinespace
\cite{lugaresi_automated_2023} & Automatic discovers simulation models from event logs. & \checkmark & \checkmark & -- & -- & -- \\ \addlinespace
\cite{castiglione_automated_2024} & Event-centric PM for generating automated digital twins?S??s. & \checkmark & \checkmark & -- & -- & -- \\ \addlinespace
\cite{belina_ethospenalps_2024} & Open-source tool for load profile simulation. & \checkmark & -- & \checkmark & \checkmark & -- \\ \addlinespace
\cite{worrlein_using_2024} & Predict energy profiles from a discrete simulation. & \checkmark & -- & \checkmark & \checkmark & \checkmark \\ \addlinespace
\cite{khodadadi_data-driven_2024} & PM extracting stochastic nets that also calculates total energy consumption. & \checkmark & \checkmark & \checkmark & -- & -- \\ \addlinespace
\cite{gonzalez_process_2025} & PM for creating process model and synthetic energy profiles from punctual energy data. & -- & \checkmark & \checkmark & -- & -- \\ \addlinespace
\cite{khodadadi_automated_2026} & Multi-flow PM for total energy and waste scalar predictions. & \checkmark & \checkmark & \checkmark & -- & -- \\ \midrule
\textbf{Our appoach} & \textbf{Extracts process model with Process mining and energy models with Machine Learning. These models form a Process and Energy Digital Twin that can be use for understanding and simulation.} & \textbf{\checkmark} & \textbf{\checkmark} & \textbf{\checkmark} & \textbf{\checkmark} & \textbf{\checkmark} \\ \bottomrule
\end{tabular}%
}
\label{realted_work}
\end{table*}

This study extends prior research by introducing a framework to extract process and energy models directly from data, constructing a Process and Energy Digital Twin for manufacturing. To achieve this, we incorporate historical process and energy data, production plans, and external factors. The main contributions of this work are: (1) manufacturing-specific considerations for automated process and energy model extraction; (2) an algorithm to extract process models with an integrated energy perspective; (3) an algorithm to extract dynamic energy profiles; (4) an algorithm to couple and integrate these models for simulation, ensuring process and energy dynamics interact and constrain one another to accurately reflect the real-world system; and (5) methods for visualization, analysis, and decision support based on the proposed models.

The resulting Digital Twin also enables a deeper understanding of process and energy behavior while supporting accurate process simulation. These capabilities enhance decision-making for process and energy optimization in industrial systems, thereby improving overall operational performance and sustainability. The proposed approach is rigorously evaluated using extensive simulated and real-world industrial data.





\section{Methodology}
\label{sec:methodology}
\subsection{Framework}

Figure~\ref{fig:framework} presents the proposed framework for automatically deriving integrated process and energy models from real-world production data. Within this framework, industrial operations and energy profiles are conceptualized through causal dependencies: the execution of the process is driven by the production plan and external factors, while the resulting energy behaviour is driven by the process events and also external factors. The external factors are exogenous variables, such as weather conditions, seasonality (e.g., month or day of the week), personnel availability, energy tariffs, and machine wear—encompass any external influences not directly captured by the event logs or energy sensor measurements. The framework to model the process an energy is structured into four phases:

\begin{figure*}[H]
    \centering
    \includegraphics[width=1\textwidth]{1_framework.pdf}
    \caption{Overview of the proposed Process and Energy Digital Twin building.}
    \label{fig:framework}
\end{figure*}

First, in \textit{Data Collection}, we gather the Produciton plan, External factors, Prcoess data and Energy data form the different IT systems of manufacturing.

Second, the production plan, process data and external factors goes into \textit{Process Model Extraction}, where it is preprocesses and then PM methdos are used to obtain the process model. Simultaneously, process data, energy data, and external factors are fed into the \textit{Energy Model Extraction} phase for preprocessing. During this stage, ML methods are applied to derive the energy models.

%Third, two parallel activities are conducted. In \textit{Process model extraction}, a process model is derived from the event log data and external factors, describing the material flow---how products move through the production system---and the machine states, representing the different operational conditions during production. Since the product flow and the machine states are the primary driver of energy consumption, this process model forms the structural backbone for energy profiling. The resulting models are formalized as Petri nets.

%In parallel, \textit{Energy model extraction} derives energy profiles from sensor data. The energy measurements are transformed and modelled using machine learning techniques to capture the relationship between process characteristics, external factors and their corresponding energy profiles.

These models are to form the Process and Energy Digital Twin. They can be used inside a decision support system to provide the insights and understanding of the process and energy, and can be used to simulate new process and energy data in scenarios or what-if analysis. In general, the Digital Twin and its simulation allows descriptive, diagnostic, predictive, and prescriptive analytics.

In the following subsections we present a more detail description of the different part of the framework.

\subsection{Data collection}

In manufacturing, process data can originate from a variety of sources. Common systems include Enterprise Resource Planning (ERP), Manufacturing Execution Systems (MES), and Supervisory Control and Data Acquisition (SCADA) for process monitoring and control, as well as local data stored in Programmable Logic Controllers (PLCs) at the machine level. These logs capture both the movement of products in the form of a material flow, such as their arrival at or departure from a machine and the different machine states involved in transforming a product.

Process data often requires aligning and combining event data from multiple sources. This may involve synchronizing time formats, removing duplicate records, and ensuring correct chronological ordering of events. These preparation steps are essential for creating a coherent event log that accurately reflects actual process behaviour and can be effectively used for subsequent process discovery and model extraction.

Energy data, on the other hand, is typically obtained from Energy Management Systems (EMS) as well as from SCADA systems or PLCs. Both event log data and energy data are often continuously measured and stored, as they are required for process traceability, regulatory compliance, energy management, and process control.

For energy data, preprocessing steps typically include verifying measurement units, identifying and correcting implausible values, and ensuring the proper calibration and reliability of sensors. In some cases, energy data must also be aggregated, as it is often recorded at very high temporal resolutions (e.g., milliseconds) for process control purposes, which may exceed the level of detail needed for process modelling.

Data for the external factors can be gathered from company IT systems or external weather services. Because their impact varies widely, such as a heating process of outside ambient air demanding more energy in winter, the relevant factors must be identified individually for each process. Furthermore, the factors affecting energy often differ from those affecting the process flow; weather, for example, typically impacts energy but not the process.

\subsection{Process model extraction}

Figure \ref{fig:process_model_extraction} shows the workflow for process model extraction. The approach uses three main data sources: process data, production plan and external factors. In the upper part, the data can be visualized and prepared either at the machine level or at the material-flow level. The data is then integrated into a unified Process Table (PT), which used by Process Mining to extract a \textit{proces model}.

\begin{figure*}[H]
    \centering
    \includegraphics[width=1\textwidth]{2_process_model_extraction.pdf}
    \caption{Process model extraction.}
    \label{fig:process_model_extraction}
\end{figure*}

Most manufacturing process modelling approaches focus on material flow \citep{camargo_discovering_2021, kouki_input_2017, rozinat_discovering_2009, castiglione_automated_2024}, while others model material flow and machine states separately \citep{friederich_data-driven_2022, friederich_framework_2022, friederich_process_2022, gonzalez_process_2025}. For energy-profile modelling, however, both perspectives must be linked: a production plan starts the material flow and it determines the routing of products through the production system and when machines are triggered to process them, with the corresponding events recorded in a \textit{material-flow event log}; the starting of the machine and product characteristics in turn affect machine activities, their internal operational modes or states of the equipment during work, which are recorded in a \textit{machine event log}. The machine´s activities define later the energy profiles.

The two types of event logs can be obtained from different IT systems (e.g., MES and PLCs, respectively) or jointly from a single system (e.g., SCADA), but it is important to keep the two types of events logically separate. Since visualizing both perspectives together can produce complex ``spaghetti-like'' models \citep{castiglione_automated_2024}, we propose using them separately for \textit{visualization} but integrating them for \textit{modelling}.

We illustrate this approach using a hospital infusion-bag production line, shown in Figure \ref{fig:process_model_extraction}. Here, the bags are filled, sterilized and packaged. The system's behavior is captured using the two types of event logs. First, a \textit{material-flow event log} tracks the movement of batches as they enter and leave the filling, sterilization, and packaging machines. Second, a \textit{machine event log} records the sterilization oprations decomposed into preparation, heating and cooling activities, each with a characteristic energy profile. For \textit{visualization}, to maintain readability, process discovery is executed independently for the material flow and for each machine's state log, yielding the distinct Directly-Follows Graphs (DFGs) depicted below their respective tables.

For the \textit{modelling} of the process and its energy behaviour, the level of detail must match the granularity of the underlying energy profiles. Each profile is determined by the execution of a machine activity on a given product and external factors at that moment. This would result in a better modelling that when considering a material-flow activity, which can have inside several machine activities.

Figure~\ref{fig:machine_activities} illustrates this point. It shows a single material-flow activity that actually consists of three distinct machine activities, each with its own characteristic energy signature that recurs whenever that activity is executed. Because the sequence of these machine activities can vary — which directly alters the overall energy profile — modelling solely at the material-flow activity would not allow accurate prediction of the energy profile (i.e., a change in the internal machine activities would produce a different profile).

Additionally, in existing process modelling approaches that inconsider energy, the durations are typically obtained by fitting a statistical distribution to observed activity durations, which is then used for simulation \citep{khodadadi_automated_2026}. However in our case, if the goal is to predict a complete energy curve, duration has a direct and substantial impact on the resulting energy profile. We therefore propose that durations can be predicted with ML models, since they can capture better the underlying variability than a purely distribution fit. These algorithms can use any input variables such as activity type, process attributes, and external factors to predict achieve a better prediction of the durations.

Also, an important point is that PM for process modelling and simulation is most commonly applied in discrete manufacturing environments, where clearly defined case\_ids exist for individual product units, such as the infusion bags in the example above, household appliances, cars on a production line, or mobile devices \citep{friederich_data-driven_2022, camargo_discovering_2021}. In these contexts, material flows are easy to monitor and track, supporting detailed product traceability. Many energy-intensive industries, however, operate continuous processes, as in chemical manufacturing or food production, which are designed for high resource utilization and stable production flows \citep{kemp_pinch_2007}. In such cases, a production order can serve as a case\_id, grouping products that share characteristics such as a common recipe \citep{gonzalez_process_2025}. The material flow can then often be reduced to a single activity, even though the underlying machine may still execute several distinct activities. It is important to note that although material flow is continuous, the underlying machine activities remain discrete — they can be interpreted as process activities with start and end times and modelled accordingly.

\begin{center}
    \includegraphics[width=1\columnwidth]{4_machine_activites.pdf}
    \captionof{figure}{Material flow and machine activities. The profile shows different characteristic behaviours across the machine states.}
    \label{fig:machine_activities}
\end{center}

\paragraph{Process model extraction algorithm.} Considering the ideas described before about the material-flow activites, machine acitvies and durtations and their relevance for \textit{modelling}, the extraction and training pipeline is summarized in Algorithm~\ref{alg:pm_extraction_ml}.

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

In Step~(A), the material-flow and machine-activity event logs are merged and flattened at the lowest available granularity. Where machine activities are recorded, as during sterilization, they are used; where they are not, as during filling, the material-flow activity is used instead. Each activity instance is then enriched with case attributes and external factors, such as time of day or ambient conditions, forming the Process Table (PT).

Step~(B) applies process discovery to extract the control flow, independently of the chosen mining algorithm. Rather than fixing one miner and one parameter setting, every miner is run over a grid of configurations (the Alpha miner, which has no tunable parameters, runs in its default setting), and each discovered Petri net is scored on four standard quality criteria: fitness, how much of the observed behaviour the model can replay; precision, how much behaviour it allows that never occurs in the log; generalizability, whether it also explains valid unobserved behaviour instead of overfitting the training log; and simplicity, how compact and interpretable its structure remains \citep{buijs_quality_2014}. The four criteria are averaged into one score, and the best configuration of each miner is retained.

Step~(C) assigns a duration to every transition of the Petri net, from two alternative sources. The statistical one fits parametric families to the observed durations of each activity by maximum likelihood and keeps the best-fitting family. The predictive one trains machine learning models to estimate the duration of a single activity instance from case attributes, external factors and process-level features such as the position of the activity in the case, the elapsed case time and the recent lagged durations. Two strategies are compared: a \textit{local} one, with a model per activity, and a \textit{global} one, with a single model over all activities that uses the activity identity as a feature. In both, the model with the lowest cross-validated mean absolute error is retained. Since durations are right-skewed, such a model tracks the median rather than the mean, so it is rescaled by the ratio between the observed mean duration and its own out-of-fold mean prediction. Activities with too few instances fall back to the median duration of the activity, and non-positive predictions fall back to the fitted statistical distribution.

Step~(D) completes the stochastic Petri net and adds the case-level budget. Each transition receives a stochastic weight given by the number of times it fires when the training log is replayed on the net; transitions that are never replayed keep a small residual weight, so that unobserved behaviour remains possible. Where several transitions compete at a decision point, each is chosen with a probability proportional to its weight. For every activity, two further distributions are recorded from the log: how often it occurs within a case, and how long the process waits before it starts. The first keeps the number of repetitions of a simulated case in a realistic range, instead of leaving loops to an unbounded random choice at every visit; the second reproduces the idle time between consecutive activities. Finally, a regression model predicts the total duration of a case from its case attributes, and is kept only if it beats the median case duration on held-out cases; otherwise the median is used.

The algorithm returns two models. The first is the stochastic Petri net, which describes the control flow, the probability of each routing decision and the repetition and waiting behaviour of the cases, and to whose transitions the duration models are attached. The second is the case-duration model, which gives the time budget a simulated case is generated to fill. Both are fed with a production plan and external factors to generate process data that reflect the control flow, the operational constraints and the temporal dynamics of the real process, and that are therefore also suitable for energy modelling.


\subsection{Energy model extraction}

The concept of energy model extraction is illustrated in Figure~\ref{fig:energy_model_extraction}. The approach uses three data sources: process data, energy data, and external factors. The energy data is split into energy profiles are transformed and enriched with process and external data. This forms an Energy Table (ET), which is used my ML algorithms to extract the \textit{Energy models}.

\begin{figure*}[H]
    \centering
    \includegraphics[width=1\textwidth]{3_energy_models_extraction.pdf}
    \caption{Energy model extraction.}
    \label{fig:energy_model_extraction}
\end{figure*}

Sensor´s measurements are typically recorded as continuous time series of timestamps and values at regular intervals (e.g., once per second), as shown in the first two columns of Table on the left of Figure~\ref{fig:energy_model_extraction}. To extract activity-specific energy profiles, these measurements must be linked to process information by the corresponding activity and case\_id. This linkage is achieved either directly—when process monitoring systems (e.g., SCADA) store sensor and production data together—or indirectly based on time. In the latter case, isolated sensor data is joined with separate process data (e.g., from MES or ERP) by assigning to each measurement the case\_id and activity that is being executed at that time.

Energy profiles in industrial processes often show recurring patterns per process activities, but these patterns are not identical across executions. Their values, peak and timing can vary due to the activitie´s duration, product characteristics, previous activities or external factors. Following the example of infusions bags, the first production batch may require additional heating to warm up the machines,larger infusion bags may need more heat for sterilization, and preparation steps may be skipped when several orders of the same product are processed consecutively. As a result, a direct point-wise patter extracton between two curves is not suitable. Thus, we aim with data transformation to capture both the general shape of a profile and its individual characteristics into a canonical or standard timeline.

In Figure~\ref{fig:energy_model_extraction}, this is shown by two curves representing two executions of the same activity. Both curves are characterized by a rapid increase followed by a decrease, which is typical for processes such as steam demand for heating in a sterilizer. Although the overall pattern is similar, the two curves differ slightly due to variations between individual executions, for instacne in total duration, maginute of the values and timing of the peak.

In our case, we want to predict a dynamic energy profiles from discrete process information, like activities and product characteristics. Hence, the complete energy profile must be modelled from a discrete input rather than a single total predicted value or a time-serie per step prediciton. The prediction problem can be described as learning a mapping from an input descriptor $\mathbf{x}$, such as a production order, to a complete time-dependent output curve:
\begin{equation}
\hat{\mathbf{y}} =
\begin{bmatrix}
\hat{y}_{1} \\
\hat{y}_{2} \\
\vdots \\
\hat{y}_{T}
\end{bmatrix}
= f(\mathbf{x}) + \boldsymbol{\varepsilon},
\end{equation}
where $f(\mathbf{x})$ denotes the model applied to the input vector $\mathbf{x}$, $\hat{\mathbf{y}} \in \mathbb{R}^{T}$ denotes the predicted energy profile over a forecasting horizon of $T$ time steps, and $\boldsymbol{\varepsilon} \in \mathbb{R}^{T}$ is the corresponding error vector. In this formulation, time is not provided explicitly as an input to the model; instead, the model must generate the complete output curve directly from the discrete input.

\paragraph{Energy model extraction algorithm.} Considering the ideas described before about the individual energy profiles patterns per activity, we propose the Algorithm~\ref{algorithm:energy_model_extarction} for its extraction. This results in the \textit{energy models}. 

In Step~(A), for each combination of case\_id and activity, the energy signal is split into individual energy profiles, or curves $y_i$. Since these individual profiles may have different lengths, Step~(B) resamples them to a fixed canonical length $S$ and computes a representative reference curve $r$ using DTW Barycenter Averaging (DBA) \citep{petitjean_global_2011, schultz_nonsmooth_2018}. This reference curve defines a common canonical timeline for the profile of the activity. The length $S$ is the median length of the observed profiles of that activity, so each activity obtains a canonical timeline with the resolution of a typical execution.

DBA differs from a plain average of the curves. A plain mean averages the profiles index by index, so a peak that occurs early in one execution and late in another is averaged against flat sections, and the resulting curve becomes flattened and too wide. DBA instead alternates two steps: every curve is aligned to the current reference by DTW, and the values matched to each canonical index are averaged into a new reference. After a few iterations, the recurring features keep their shape and their height, because peaks are only averaged with peaks. In our implementation the matched values are aggregated with a trimmed mean, so that a few executions whose duration differs strongly from $S$ cannot distort the reference shape.

In Step~(C), each curve is aligned to the reference curve using Dynamic Time Warping (DTW) \citep{giorgino_computing_2009}. DTW compares time series with similar shapes but different temporal progressions by allowing local stretching and compression of the time axis. Thus, similar segments can be matched even if they occur earlier, later, faster, or slower in different executions of the activity. This makes it possible to align corresponding activity phases consistently across executions. The result is an aligned curve $\tilde{y}_i$ of fixed length $S$.

\begin{algorithm}[H]
\caption{Energy profile extraction.}
\label{algorithm:energy_model_extarction}
\begin{algorithmic}[1]
\Require Table $\mathcal{D}$ with case, activity, timestamps, attributes $\mathbf{x}^{\mathrm{attr}}$, external features $\mathbf{x}^{\mathrm{ef}}$ and energy values; regressors $\mathcal{R}$

\State \textbf{(A) Extract activity-level energy profiles}
\State For each pair of case $c$ and activity $a$ in $\mathcal{D}$, sort its rows by timestamp and extract their energy values as a raw curve $y_i$ of length $|y_i|$, with the process features $\mathbf{z}_i$ given by $\mathbf{x}^{\mathrm{attr}}$ and $\mathbf{x}^{\mathrm{ef}}$

\State \textbf{(B) Build the reference curve}
\State Set the number of latent positions $S$ to the median length of the raw curves of the activity, and resample every raw curve $y_i$ to that length
\State Compute the reference curve $r$ from the resampled curves by DTW Barycenter Averaging, trimming the most deviating curves at each iteration so that unusually long or short instances do not dominate the average.

\State \textbf{(C) Align the curves}
\For{each raw curve $y_i$}
    \State Align $y_i$ to $r$ by DTW and average the values matched to each reference position, obtaining the aligned curve $\tilde{y}_i$ of length $S$
    \State If $y_i$ is much shorter than $r$, interpolate it linearly onto the $S$ positions instead, since its warping path would collapse into a step function
\EndFor

\State \textbf{(D) Build the training table}
\For{each aligned curve $\tilde{y}_i$}
    \For{each latent position $s = 1,\dots,S$}
        \State Add to $\mathcal{E}$ one row with features $(\mathbf{z}_i,\, a,\, s,\, s/S,\, |y_i|)$ and target $\tilde{y}_i[s]$
    \EndFor
\EndFor
\State Encode the categorical variables and keep the numerical variables unchanged

\State \textbf{(E) Train the curve model}
\State Fit each $h \in \mathcal{R}$ on $\mathcal{E}$, optionally tuning its hyperparameters, and keep as $g$ the model with the lowest validation error

\State \Return the curve model $g$, the reference curve $r$ and the number of latent positions $S$, for each sensor, and activity
\end{algorithmic}
\end{algorithm}

\begin{figure*}[H]
    \centering
    \includegraphics[width=1\textwidth]{5_dba_dtw_alignment.pdf}
    \caption{Curve transformation. (a) Original curves with different lengths and different internal event timing; (b) Curves after linear resampling to a common 100-point representation; timing differences remain; (c) Curves after DTW alignment to the DBA barycenter; timing is corrected while value differences remain.}
    \label{fig:dba_dtw_alignment}
\end{figure*}

\begin{figure*}[H]
    \centering
    \includegraphics[width=0.7\textwidth]{6_simulation.pdf}
    \caption{Process and energy simualtion.}
    \label{fig:simualtion}
\end{figure*}

An example of curve resampling, DTW Barycenter Averaging, and DTW alignment is shown in Figure~\ref{fig:dba_dtw_alignment}. Visual (a) shows the original curves from multiple executions of a machine's activity, each with different lengths, shape timings, and values. Visual (b) shows the curves resampled to a fixed length; their overall forms are similar, but local shapes like peaks and valleys still occur earlier or later. Visual (c) shows the curves after DTW alignment to the DBA barycenter, where the shapes are temporally aligned while the differences in the energy values remain.

In Step~(D), each aligned curve is added to the Energy Table (ET), with one row per canonical index $s$. Every row carries that index, its relative position $s/S$ within the curve, the original length of the curve, and the corresponding process information, which is repeated for every row belonging to the same curve. We also add the external factors at that point in time that might affect the energy demand, for instance the ambient temperature.

In Step~(E), the regression algorithms $\mathcal{R}$ are trained on the ET. This allows them to learn how the value at every canonical index of the transformed energy profile is associated with the activity, the product attributes, and the external factors. The models are evaluated with cross-validation, and we select the model $g$ with the best evaluation metric.

Overall, the combination of resampling, DTW-based alignment, and ML modelling allows the prediction of realistic energy profiles while considering process-related and external information.





\subsection{Decision Support System}

In this section we make a closer forcus on the decision suport system, epcifically in the simualtion and in the posibilites for deiciosn suporot that the digital twin and it simualtion offer.

Figure~\ref{fig:simualtion} shows the steps of the simulation stage. Here the digital twin, with the process models learned before, is used to generate new process data and the corresponding energy data. First, the new production plan and the expected external factors are fed to the process simulation, which unfolds each planned case into a sequence of activities with their simulated start times and durations. The simulated events are then joined with the external factors, and the resulting table is used to predict the energy profile of every activity instance. These profiles are predicted on the canonical timeline and are finally transformed back to the simulated duration of each activity, so that the energy demand follows the timing of the simulated process.


\begin{algorithm}
\caption{Simulation of the process and energy.}
\label{alg:simulation}
\begin{algorithmic}[1]
\Require Stochastic Petri net $\mathcal{N} = (N, M_0, M_f, \mathcal{G}, w)$ with the case-behaviour distributions $R_a$ and $W_a$, and case-duration model $\hat{f}_B$, from Algorithm~\ref{alg:pm_extraction_ml}; curve models $(g, r, S)$ from Algorithm~\ref{algorithm:energy_model_extarction}; production plan $P$ with the attributes $\mathbf{x}^{\mathrm{attr}}_c$ of every planned case and the external factors $\mathbf{x}^{\mathrm{ef}}$; exit weights $\alpha < 1 < \beta$; step limit $k_{\max}$

\State \textbf{(A) Initialise the case}
\State Take the row of case $c$ from the production plan $P$ and assemble its feature vector $\mathbf{z}^\star$ from $\mathbf{x}^{\mathrm{attr}}_c$ and $\mathbf{x}^{\mathrm{ef}}$
\State Predict the time budget of the case, $B^\star = \hat{f}_B(\mathbf{x}^{\mathrm{attr}}_c)$
\State Set the marking $M \gets M_0$, the elapsed case time $\Delta \gets 0$, the step counter $k \gets 0$ and the fire counters $\nu_a \gets 0$ for every activity $a$
\State Draw the repeat quota of this case, $\kappa_a \sim R_a$, for every activity $a$

\State \textbf{(B) Replay the Petri net}
\While{$M \neq M_f$, $\Delta < B^\star$ and $k < k_{\max}$}
    \State Determine the transitions enabled under $M$ and take their weights from $w$
    \State Discount the weight of every enabled transition whose activity has already reached its quota, $\nu_a \geq \kappa_a$, as a soft penalty rather than a hard block
    \State Multiply by $\alpha$ the weight of the transitions that would lead to the final marking $M_f$, so that the case keeps generating activities while its budget is not spent
    \State Sample the transition $t^\star$ with probability proportional to its weight and fire it, $M \gets \mathrm{fire}(M, t^\star)$
    \If{$\ell(t^\star) = a$ is a visible activity}
        \State Sample its duration $d^\star$ from $D_{t^\star} \in \mathcal{G}$ and the idle time preceding it, $\delta \sim W_a$
        \State Append the instance $(a,\, \Delta + \delta,\, d^\star)$ to the case log $\mathcal{L}$ and update $\Delta \gets \Delta + \delta + d^\star$ and $\nu_a \gets \nu_a + 1$
    \EndIf
    \State $k \gets k+1$
\EndWhile
\State Once the budget is spent, multiply by $\beta$ the weight of the transitions leading to $M_f$ and replay until the final marking is reached, so that the case closes on a valid end

\State \textbf{(C) Predict and place the energy profiles}
\For{each activity instance $(a, \Delta_a, d^\star) \in \mathcal{L}$ and each sensor}
    \State Predict its energy profile $\hat{y}^\star$ of length $d^\star$ with the models $(g, r, S)$ of that sensor and activity from Algorithm~\ref{algorithm:energy_model_extarction}, using the features $\mathbf{z}^\star$ and the duration $d^\star$
    \State Insert $\hat{y}^\star$ into the simulated energy timeline $\hat{Y}$ over the interval $[\Delta_a,\, \Delta_a + d^\star]$
\EndFor

\State \Return the simulated case log $\mathcal{L}$ and the energy timeline $\hat{Y}$, repeating (A)--(C) for every case of the production plan $P$ and superposing the resulting timelines
\end{algorithmic}
\end{algorithm}


\paragraph{Simulation of the process and energy.} Considering the ideas described before about the material-flow activites, machine acitvies and durtations and their relevance for \textit{modelling}, the extraction and training pipeline is summarized in Algorithm~\ref{alg:simulation}.

In Step~(A), the simulation takes one case of the production plan and builds its feature vector from the case attributes and the external factors expected at that time. The case-duration model then predicts the time budget of the case, that is, how long the case should take in total. The Petri net is set to its initial marking, the elapsed case time is set to zero, and a repeat quota is drawn for every activity from the distribution of repetitions per case observed in the training log. This quota fixes, for this particular case, how often each activity may occur, so that loops in the net produce realistic repetitions instead of an unbounded random choice at every visit.

Step~(B) replays the Petri net. At each step the enabled transitions are determined and weighted with the routing probabilities of the process model. Two corrections are applied to these weights. First, a transition whose activity has already reached its quota is strongly discounted, but not blocked, so that the case can still move forward when no alternative exists. Second, the transitions that would end the case are discounted while the predicted budget is not yet spent, which keeps the case generating activities until it reaches a realistic total duration. A transition is then sampled according to the corrected weights and fired. If it corresponds to a visible activity, its duration is drawn from the firing time attached to that transition, the idle time preceding it is drawn from the waiting distribution of the activity, and the resulting instance is appended to the simulated case log. The elapsed case time advances by the idle time and the duration, so the budget is consumed by the activities themselves and no timeline is rescaled afterwards. Once the budget is spent, the transitions that end the case are favoured instead of discouraged, and the replay continues only until the final marking is reached, so that the case always closes on a structurally valid end.

In Step~(C), each simulated activity instance receives its energy profile. The curve model of the corresponding sensor and activity predicts the profile on the canonical timeline and maps it back to the duration of that case. The profile is then placed on the simulated timeline over the interval covered by the instance, so that the energy demand follows the same control flow and the same timing as the simulated process.

Steps~(A) to~(C) are repeated for every case of the production plan, and the resulting profiles are superposed into a single energy timeline. The simulation therefore produces both a synthetic event log, with realistic activity sequences and durations, and the associated energy demand, from nothing more than a production plan and the external factors expected for it.

\paragraph{Analytics.} The proposed process-energy digital twin supports decision-making at the descriptive, diagnostic, predictive and prescriptive level, since it links the energy demand to the process that causes it and can be run forward on a planned scenario. Table~\ref{table:analytics_levels} gives examples of the analyses enabled at each level.


\begin{table}[H]
\caption{Example of analytics offered by the Process and Energy Digital Twin.}
\label{table:analytics_levels}
\begin{tabular}{p{1.4cm}p{6cm}}
\toprule
Type & Examples \\
\midrule
Descriptive &
$\bullet$ Reconstruction and visualization of process-linked energy profiles \newline
$\bullet$ Assignment of energy data to activities, cases, machines, and products \newline
$\bullet$ Characterization of historical energy behavior
\\[0.5em]
Diagnostic &
$\bullet$ Explanation of why energy profiles differ across executions \newline
$\bullet$ Identification of the influence of duration, product, disturbances, and operational conditions \newline
$\bullet$ Detection of abnormal relationships between process execution and energy demand
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

\subsection{Algorithms for the implementation}

The implementation of the methods and experiments is carried out in the Python programming language. For the machine learning algorithms used to model the energy profiles and predict activity durations, we employ standard implementations of Linear Regression, Multilayer Perceptron Regressor, and Feed-Forward Neural Network Regressor from the scikit-learn library \citep{pedregosa_scikit-learn_2011}. Also, we use a Extreme Gradient Boosting Regressor from the XGBoost library \citep{chen_xgboost_2016}. For curve prediction only, we additionally use as Seq2Seq ML algorithms an Long Short Term Memory (LSTM) neural netowrk and a Transformer implemented in PyTorch \citep{paszke_pytorch_2019}. For hyperparameter tuning of the ML algorithms we use Optuna \citep{ozaki_optunahub_2025}.

For the PM part, we use three process discovery algorithms — the Alpha Miner \citep{van_der_aalst_workflow_2004}, the Heuristic Miner \citep{weijters_flexible_2011, weijters_process_2006}, and the Inductive Miner \citep{leemans_discovering_2013} — as implemented in the PM4Py library \citep{berti_pm4py_2023}.For curve modelling, we use the Dynamic Time Warping (DTW) algorithm \citep{giorgino_computing_2009} and the DTW Barycenter Averaging (DBA) algorithm from the tslearn library \citep{tavenard_tslearn_2020}. Distribution fitting (Normal, Lognormal, Exponetial, Gamma) is used from the scipy library \citep{virtanen_scipy_2020}.

\subsubsection{Evaluation protocol}

The evaluation´s logic can be seen in Figure~\ref{fig:evaluation_protocol}. For the evaluation, we first apply a temporal split to the data, using 70\% for training and 30\% for testing to avoid data lackage. The model extraction is performed using the training data, which corresponds to the earlier portion of the timeline, while the evaluation metrics are computed on the test data, corresponding to the later portion. This setup prevents temporal leakage and ensures that the models are assessed on future, unseen data.

We propose three evaluations to properly assess the different components of the proposed framework with empirical evidence. The results is divided divided into four parts, first to demonstrate the visualization of the process and energy combined, second the evaluation of individual energy curves, third the evaluation of the rpoces and foruth the evlaution fo the process and energy combined. The evalautions are explained in more detail in the following subsections in conjuntion with their relevant metrics.

\begin{center}
    \includegraphics[width=1\columnwidth]{7_evaluation_protocol.pdf}
    \captionof{figure}{Evaluation protocol.}
    \label{fig:evaluation_protocol}
\end{center}


\subsection{Evaluation 1: process.}
\label{sec:complete-curve-eval}

This evaluation assesses the quality of the fully generated process and energy behaviour for complete traces. Starting from the test production plan and the external factors, we simulate new process instances and predict their energy profiles. A direct comparison with the test set is challenging, since small deviations in the executed sequence can strongly shift the position of the energy profiles, which makes a point-wise comparison with the real test profiles unreliable.

We therefore first evaluate whether the methods are able to reproduce the structure of the process. Three process models are considered: \textit{Alpha}, the Alpha miner used as a baseline; \textit{Best Petri net}, the best discovered Petri net; and \textit{Best Petri net + Budget}, the best Petri net with duration budgeting at simulation time. Each of them is crossed with the three duration predictors: \textit{baseline}, which samples from the fitted statistical distributions; \textit{ml\_local}, one machine learning model per activity; and \textit{ml\_global}, a single model for all activities. This results in nine combinations.

The structure of the discovered model is assessed with four conformance metrics, obtained by replaying the real test sub-log of each station on the corresponding Petri net, so that none of them involves the simulated log. \textbf{Fitness} measures how completely the model reproduces the observed behaviour, penalising missing and remaining tokens during replay \citep{rozinat_conformance_2008}. \textbf{Precision} measures how selectively it constrains behaviour, penalising models that enable activities never observed at that point in the log \citep{munoz-gama_fresh_2010}. \textbf{Generalization} rewards models whose transitions are visited often during replay, since structures seen only once or twice are likely overfitted to the training log, and \textbf{Simplicity} penalises structurally complex nets, whose places and transitions carry many arcs \citep{buijs_quality_2014}. All four range from 0 to 1, with 1 as the best value, and are partly conflicting, since a model can always be made more fitting by allowing more behaviour, so they are also averaged into a single score.

The simulated log itself is evaluated with the \textbf{Evt-Ratio Error}, the ratio between the number of simulated and real events, and with the temporal characteristics of the process, namely the duration of the activities and the span of a case, that is, its cycle time. Both are assessed with \textbf{MAE} and \textbf{WAPE} and, in contrast to the energy profile evaluation, are computed on the original, non-standardised values and reported in minutes.

\subsection{Evaluation 2: individual energy profiles}

The goal of this evaluation is to assess the effectiveness of the proposed energy-modelling method at the activity level. To this end, we use the process instances of the test set, with their activity and their relevant attributes, together with the external factors, such as weather variables and the day of the week, to predict the energy profiles with the proposed approaches. The predicted curves are then compared with the corresponding ground-truth profiles of the test set.

The comparison is complemented by an ablation study, in which individual components of the methodology are removed, so that the contribution of each component can be quantified and the design decisions can be supported not only by the literature but also by empirical evidence. The components varied are the use of DTW, the type of model, either a regression model or a Seq2Seq model, and the use of the external factors. We further include the \textit{median per activity and sensor}, as used by \citet{gonzalez_process_2025}, whose approach corresponds to the combination \textit{DTW + Seq2Seq}, and a \textit{baseline} given by the naive median value of the sensor. This results in eight methods: \textit{baseline}, \textit{median per activity and sensor}, \textit{ML only}, \textit{Seq2Seq only}, \textit{DTW + ML}, \textit{DTW + Seq2Seq}, \textit{DTW + ML + Ext. Factors} and \textit{DTW + Seq2Seq + Ext. Factors}, where \textit{ML only} and \textit{Seq2Seq only} are trained without DTW alignment.

Since the goal is to assess how realistically the predicted profiles reproduce the behaviour of the real ones rather than their point-wise accuracy, the quality of the predictions is measured with five statistical metrics, each computed on the predicted and the ground-truth curve and compared between the two: \textbf{Sum}, the total energy of the curve; \textbf{Max}, the peak value of the curve; \textbf{Mean} and \textbf{Std}, the mean and the standard deviation of the curve; and \textbf{Roughness}, which quantifies the point-to-point variability of the curve and thus penalises predictions that are unrealistically smooth.

The deviations are normalized by the standard deviation of the ground-truth test data of the corresponding sensor, making results comparable across sensors of different physical scales

%Both the predicted and the ground-truth curves are standardised with the mean $\mu$ and the standard deviation $\sigma$ of the ground-truth test curve. Using the same statistics for both makes the results comparable across sensors with different physical scales, while the predicted profile is still evaluated on the scale of the real data. All metrics are computed on the standardised values, are non-negative, and take the value $0$ for a perfect prediction.


\paragraph{Evaluation 3: process and energy}

Assessing the complete energy profile of a case is more challenging than assessing individual activity curves. A point-by-point comparison is not meaningful here, because small variations in the simulated execution shift the predicted profiles in time, so that they can no longer be aligned with the profiles observed in the real test data.

For this evaluation we combine the process models of the previous section, \textit{Alpha Petri net}, \textit{Best Petri net} and \textit{Best Petri net + Budget}, each with its best duration prediction (\textit{ml\_local}, \textit{ml\_global} or the stochastic durations), and with the best curve model of the first energy evaluation. In addition, we include three approaches that do not rely on the full process model, in order to assess how much the process modelling contributes to the quality of the energy prediction: a \textit{Median baseline}, which uses the median energy curve; a \textit{Profile-generator}, which generates the energy profile of a case from a learned case-level energy curve; and \textit{Schedule-direct}, which uses the best curve model of the first energy evaluation but learns and predicts the whole profile of a case directly from the production schedule, skipping the process modelling and the simulation.

The evaluation is carried out per case, that is, per order of the production schedule, and per sensor, comparing the \emph{complete} energy profile of each test case with its simulated counterpart instead of isolated per-activity curves or population-level statistics. Each real test case $c$ is matched with the simulated case of the same identifier, since the simulation preserves the case identity even when the generated activity sequence differs. For a case $c$ and a sensor $s$, the \emph{real profile} consists of the measured values of $c$ ordered by timestamp, and the \emph{predicted profile} concatenates, in the same order, the curves predicted for the simulated activity instances, each conditioned on its simulated duration and its metadata. In both profiles the time is measured relative to the start of the first activity of the case, so that scheduling offsets between the real and the simulated timeline do not affect the comparison.

Since a point-wise comparison is not possible, the metrics assess how well each method reproduces the statistical properties of the real profiles. We therefore use the same realism-oriented metrics as in Evaluation~2, namely \textbf{Sum}, \textbf{Max}, \textbf{Mean}, \textbf{Std} and \textbf{Roughness}, computed per case and sensor, together with \textbf{Overall}, the average of these metrics, used as a single summary value.


\subsection{Datasets}

For the evaluation, we use event logs and energy time series from six industrial processes, summarized in Table~\ref{table:datasets}.

\begin{table}[width=.9\linewidth,cols=5,pos=h]
\caption{Dataset information overview. Hours refer to the recorded length of the energy time series.}
\label{table:datasets}
\begin{tabular}{lrrrr}
\toprule
dataset & \makecell{number of\\cases} & \makecell{number of\\activities} & \makecell{number of\\sensors} & hours \\
\midrule
process\_1 & 300 & 25 & 6 & 606 \\
process\_2 & 48 & 12 & 5 & 512 \\
process\_3 & 49 & 12 & 5 & 512 \\
process\_4.1 & 160 & 5 & 18 & 2387 \\
process\_4.2 & 156 & 5 & 17 & 2463 \\
process\_5 & 53 & 20 & 28 & 501 \\
\bottomrule
\end{tabular}
\end{table}

The \texttt{process\_1} is a synthetic dataset of medical infusion bag production, covering distillation, bottling, autoclave sterilization, and packaging. Events were generated by discrete-event simulation and the electricity, steam, and cooling profiles from physical process equations, so the ground-truth link between activities and energy demand is known (Appendix~X).

The \texttt{process\_2} and \texttt{process\_3} are the two production lines of a wet-mixing area at a baby-food manufacturer, where milk is heated, homogenized, and cooled. The lines run independently but share the steam meter of the mixing.

Also, \texttt{process\_4.1} and \texttt{process\_4.2} stem from the same spray-drying tower producing milk powder, dominated by steam-based air heating. They cover the periods before and after a modification of the tower, which changed run lengths, interruption frequency, and the level of several process variables; we therefore treat them as two distinct processes.

Finally,\texttt{process\_5} is the heating network of a pasteurization plant at a fruit-juice producer, with steam-based heating and cooling-tower and chilled-water cooling. 

Across all four processes, measurements include thermal heating and cooling power, process temperatures, product recipe data, and activity logs. Weather variables — ambient temperature, relative humidity, and solar irradiance — are incorporated as external covariates for energy modelling, while day-of-week and month serve as temporal features for both process and energy modelling.

Collectively, the datasets comprise 766 cases and 79 energy time series covering close to 7{,}000 hours of operation, providing a diverse and realistic basis for evaluating the proposed methods across varying process scales and energy profiles.


\section{Results}
\label{sec:results}

\subsection{Results of evaluation 1: process}

In this subsection we show the results for the process evalaution, they are shown in Table~\ref{tab:process\_results}. It can be seen that the \textit{Best Petri net} and \textit{Best Petri net + Budget} had the same results in  Fitness,Precision, Generalization and simplicity, since they use the best process model that is given. In these metrics, the \textit{Alpha (Baseline)} had on average xxx \% worse results on those metrics.

Regarding the duration metrics, here the mehtods differ widely. First the ones that use \textit{ml\_local} had for the prediction of the Dur-act better results, on average of xxx\% than the \textit{ml\_global}, and xxx\%  bettern that the median baseline. For those metrics, the \textit{Best Petri net} and \textit{Best Petri net + Budget} were not largely different but the last was the best.

Interestingly, for the span case durations, the differente between the methods is large. Here the \textit{Best Petri net + Budget} had conside3rly better resutl than the other methods, about xxx \% than the best method of \textit{Best Petri net} and xxx \%  than the \textit{Alpha (Baseline)}. This is also the same in the case of the \textit{Evt-Ratio Error}, where the \textit{Best Petri net + Budget} is xxx \% bettern than the \textit{Best Petri net}. In these metrics, related to time, the perfromance of the \textit{Best Petri net} and \textit{Alpha (Baseline)} is poor, with large MAE and WAPE errors (above xxx WAPE).


\begin{table*}[H]
\centering
\caption{Process modelling and timing accuracy (test set results, median across processes and cases)}
\label{tab:process\_results}
\vspace{-0.5em}
\setlength{\tabcolsep}{4pt}
\renewcommand{\arraystretch}{1.3}
\footnotesize
\begin{tabular}{ll|c|c|c|c|c|c|c|c|c}
\toprule
Method & \makecell[l]{Time\\approach} & Fitness & Precision & \makecell{Generalization} & Simplicity & \makecell{Evt-Ratio\\Error} & \makecell{Dur-act\\WAPE (\%)} & \makecell{Dur-act\\MAE (min)} & \makecell{Case \\span\\WAPE (\%)} & \makecell{Case \\span MAE\\(min)} \\
\midrule
\multirow{3}{*}{\makecell[c]{Alpha \\ (Baseline)}} & baseline & 0.635 & 0.420 & 0.556 & 0.477 & 0.496 & 49.869 & 6.6 & 70.341 & 337.5 \\
 & ml\_global & 0.635 & 0.420 & 0.556 & 0.477 & 0.496 & 66.230 & 10.1 & 58.291 & 479.2 \\
 & ml\_local & 0.635 & 0.420 & 0.556 & 0.477 & 0.496 & 49.577 & 8.2 & 60.220 & 246.8 \\
\cline{1-11}
\multirow{3}{*}{\makecell[c]{Best \\ Petri net}} & baseline & \textbf{0.994} & \textbf{0.728} & \textbf{0.664} & \textbf{0.660} & 0.228 & 60.525 & 13.2 & 85.742 & 570.3 \\
 & ml\_global & \textbf{0.994} & \textbf{0.728} & \textbf{0.664} & \textbf{0.660} & 0.228 & 62.685 & 11.5 & 69.217 & 454.5 \\
 & ml\_local & \textbf{0.994} & \textbf{0.728} & \textbf{0.664} & \textbf{0.660} & 0.228 & 47.741 & 8.5 & 74.096 & 384.3 \\
\cline{1-11}
\multirow{3}{*}{\makecell[c]{Best \\ Petri net \\ + Budget}} & baseline & \textbf{0.994} & \textbf{0.728} & \textbf{0.664} & \textbf{0.660} & 0.087 & 46.772 & 10.4 & 31.688 & 256.8 \\
 & ml\_global & \textbf{0.994} & \textbf{0.728} & \textbf{0.664} & \textbf{0.660} & 0.244 & 67.999 & 9.6 & 23.068 & 91.8 \\
 & ml\_local & \textbf{0.994} & \textbf{0.728} & \textbf{0.664} & \textbf{0.660} & \textbf{0.067} & \textbf{39.492} & \textbf{5.7} & \textbf{21.453} & \textbf{73.0} \\
\bottomrule
\end{tabular}
\vspace{0.5em}
\noindent\raggedright\footnotesize

Description: Process discovery methods are Alpha (naive miner baseline), Best Petri net (best discovered Petri net), Best Petri net + Budget (best Petri net + duration budgeting at simulation time), each crossed with the three activity-duration predictors (baseline is the median, ml\_local is a ML model per activity and ml\_global is a ML model for all activities). Fitness, Precision, Generalization and Simplicity are discovery quality (higher is better); Evt-Ratio Error is the relation between total simulated events/number of real events, lower is better; and the Dur-act are the difference of the individual activities durations vs the observed in test set and Span are the same but for the trace length (lower is better). \textbf{Bold} = best per column.

\end{table*}


%Fitness/Precision/Generalization/Simplicity are model-level (real log replayed against the discovered net) and therefore identical across the simulation variants of one net.


\subsection{Results of the evaluation 2: individual energy profiles}

The results are aggregated using the median across all approaches, sensors, processes and activities, the main metric is the standardized sMAE, and results are sorted by this value. The complete resutls per approach senosr and activity (XXX in total 11.000 in total XXX) can be found individually in the online appendix in the code repository.

Table~\ref{tab:energy_results} shows the results. The best-performing approach was \textit{DTW + ML + Ext. Factors} across all metrics with 1.287 sMAE and 6.120\% WAPE. This was followed by \textit{DTW + ML} (xxx\% worse sMAE) and \textit{Median per activity and sensor} (xxx\% worse sMAE). The baseline showed an xxx\% worse sMAE than the best approach.

The \textit{Seq2Seq} performed in generall worse than the ML methods (the best, \textit{DTW + Seq2Seq + Ext. Factors}, had a sMAE xxx\% worse sMAE than the best method). Also, all the models that used the \textit{Ext. Factors} had better metrics than the ones that did not, the difference is on average of xxx\%. Furthermore, the models that did not use DTW had a poor performance, \textit{Seq2Seq only (no DTW)} being even worse than the basleine.

\begin{table*}[H]
\centering

\begin{minipage}{8.5cm}
\centering

\captionsetup{
    justification=centering,
    singlelinecheck=false,
    format=plain
}

\caption{Results for individual profile prediction. }
\label{tab:energy_results}

\vspace{-0.5em}

\begin{tabular}{p{6cm}|c|c|c}
\toprule
\textbf{Method} & \textbf{sMAE} & \textbf{sRMSE} & \textbf{WAPE (\%)} \\
\midrule
DTW + ML + Ext. Factors & \textbf{1.287} & \textbf{1.506} & \textbf{6.120} \\
DTW + ML & 1.819 & 2.044 & 11.456 \\
Median per activity and sensor & 1.838 & 2.098 & 8.573 \\
DTW + Seq2Seq + Ext. Factors & 1.875 & 2.098 & 9.871 \\
DTW + Seq2Seq & 1.918 & 2.218 & 11.867 \\
ML only (no DTW) & 2.228 & 2.466 & 12.973 \\
Baseline & 2.300 & 2.525 & 11.766 \\
Seq2Seq only (no DTW) & 2.328 & 2.583 & 12.895 \\
\bottomrule
\end{tabular}

\vspace{0.5em}

\parbox{11.5cm}{%
\footnotesize
Standardized median test results per method, across sensors and activities.
Lower values are better. Rows are sorted best-to-worst by sMAE.
\textbf{Bold} indicates the best value per metric. The baseline is the median value of the sensor.
}
\end{minipage}
\end{table*}




\subsection{Results of evaluation 3: process and energy}

In this subsection we evalaute the compelte process and energy, the idea is to get a compelte simualted energy profile per process case and comapre it with the real test case. According to the resutls of the previous ection, all the procews models are selected (\textit{Alpha (baseline), Best Petri net and Best Petri net + Budget}) with the best duration models (\textit{ml\_local}). They simualted the test production plan cases, and predict the energy profiles with the best emethod of the first evaluation (\textit{DTW + ML + Ext. Factors}). Their resutls are compare to approache that do not consider the process to rpedict the profiles (\textit{Profile-generator, Scheudle based rpedictor and median baseline}).
 
It can be seen that overall, the best was the \textit{Best Petri net + Budget}, followed by xxx and xxx. The baseline was the ebst in \textit{Mean} and \textit{Scheudle based predictor} sligly better in \textit{max}

Notably, the methdos that simualted the process have similar metrics, they only differ drastically in the \textit{sum}, being \textit{Best Petri net} xxx \% worse than \textit{Best Petri net}

\begin{table*}[H]
\centering

\begin{minipage}{13cm}
\centering

\captionsetup{
    justification=centering,
    singlelinecheck=false,
    format=plain
}

\caption{Results evalaution 3: process and energy}
\label{tab:energy_profile_complete}

\vspace{-0.5em}

\begin{tabular}{p{2.0cm}|p{3.4cm}|c|c|c|c|c}
\toprule
\textbf{Type} & \textbf{Method} & \textbf{Sum} & \textbf{Max} & \textbf{Mean} & \textbf{Std} & \textbf{Overall} \\
\midrule
\multirow{1}{*}{Baseline} & Baseline & 0.281 & 0.123 & \textbf{0.080} & 0.703 & 0.297 \\
\midrule
\multirow{3}{*}{Process model} & Best Petri net + Budget & \textbf{0.253} & 0.094 & 0.106 & \textbf{0.511} & \textbf{0.241} \\
 & Alpha Petri net & 0.476 & 0.093 & 0.104 & 0.526 & 0.300 \\
 & Best Petri net & 0.474 & 0.103 & 0.104 & 0.533 & 0.304 \\
\midrule
\multirow{2}{*}{Schedule-based} & Schedule-direct & 0.379 & \textbf{0.089} & 0.082 & 0.622 & 0.293 \\
 & Profile-generator & 0.379 & 0.189 & 0.088 & 1.600 & 0.564 \\
\bottomrule
\end{tabular}

\vspace{0.5em}

\parbox{13cm}{%
\footnotesize
\textbf{Baseline}: the median curve of each sensor, reused for every case. \textbf{Alpha Petri net}: net discovered by the alpha miner. \textbf{Best Petri net}: best discovered net per process, selected on the training split by the mean of Fitness, Precision, Generalization and Simplicity. \textbf{Best Petri net + Budget}: the same net, with each case generated to match its predicted total-duration budget. \textbf{Schedule-direct}: curves placed directly on the real schedule. \textbf{Profile-generator}: stochastic profile generator. The three Petri-net rows use the ml\_exog\_prev\_activity curve predictor. Cells are the median over (process, case, sensor) of the paired per-case relative error $|f(\mathrm{pred})-f(\mathrm{real})|/\overline{|f(\mathrm{real})|}$. Lower is better; \textbf{bold} = best per column. Overall is the average across the metric columns.
}

\end{minipage}

\end{table*}

\subsection{Visualization of the process and energy}

We use the framework to visualize the process, including material-flow activities, machine activities, and the corresponding energy profiles for each activity.

Figure~\ref{9_process_energy_visual.pdf} shows energy profiles of cases of the process\_1 and all methods. It can be seen that the methdos that use the process, have similar energy curves than the real energy curve, but they can be shiften in time compare to the real one. Alpha have also activites that did not come up in the rela case and best Petri net was similar than the best Petri net budget but these last´s profiels were closer to the real ones, also it span duration was closer to the rela ones.

In the case of the methdos that do not sue the process to build the profiles, their curves seem random, not being close to the rela process sturcutre, althought the metrics as shown in Table~\ref{tab:energy_profile_complete} might be similar to the ones that consider the process.

\begin{figure*}[H]
    \centering
    \includegraphics[width=1\textwidth]{9_process_energy_visual.pdf}
    \captionof{figure}{fdsfsdfdfsf.}
    \label{fig:total_profile}
\end{figure*}

Figure~\ref{fig:process_and_profile} shows an example of the results for process\_1, together with its historical behavior. The energy profiles are aligned with the activities and segmented according to each activity execution in order to analyze their individual shapes. In addition, the DBA Barycenter curve is calculated to obtain a representative energy profile for each activity.

This visualization supports process and energy diagnosis. For instance, it helps identify bottlenecks, recurring process behaviors, and their effects on energy profiles. Based on these insights, several process and energy-efficiency improvements can be explored. Since the DBA barycenter provides characteristic curves for the executions of each activity, these curves can be compared across activities such as heating and cooling steps to support the design of improved monitoring and analysis systems.

The DBA Barycenter curves for the heating and cooling activities show that the framework is able to capture complex process-related energy patterns. However, in the case of the holding activity, the energy demand does not present a distinctive shape and appears closer to a constant profile with noise. In such cases, a simple statistical representation, such as the mean profile, may be more appropriate than a machine-learning-based model. This decision can be analyzed and made at this stage based on the observed profile characteristics.

\begin{figure*}[H]
    \centering
    \includegraphics[width=0.6\textwidth]{7_process_and_profiles.pdf}
    \captionof{figure}{Heuristic net of process\_1 with with material flow, machine activites and energy profiles with DBA Baycenter.}
    \label{fig:process_and_profile}
\end{figure*}



\section{Discussion}
\label{Discussion}

In this article, we proposed the xxx for the integrated modelling of Process and Energy Digital Twin. This approach captures both the discrete process flow and the dynamic energy behavior of manufacturing systems using data-driven methods, enabling the simulation of new production instances to evaluate their impact on energy profiles for decision support.

Previous approaches typically model production processes and energy independently, or they have combined discrete process models with static, aggregated energy KPIs (such as total energy consumption). In contrast, our framework models production systems comprehensively, capturing their discrete and dynamic behaviors based on exogenous inputs like production schedules and external environmental factors.

Extensive experiments utilizing simulated and real-world event data and energy profiles from production environments validate our design decisions. In the following sections, we discuss our contributions to the literature, insights for practical applications, limitations of the current study, and avenues for future work.



\subsection{Main findings and interepretation}

\vspace{\baselineskip}
\textbf{Modelling continuous energy profiles from discrete activities}

Data-driven approaches to energy modelling in industrial systems have largely focused either on predicting point values, for instance, regression models for aggregated consumption, or on modelling energy time series from other temporal covariates alone CHECK \cite{he_generic_2020, zhang_data-driven_2021, mawson_deep_2020}. Generating a complete, continuous energy curve from strictly discrete inputs, such as activity executions and product attributes, is a considerably harder problem \cite{woerrlein_method_2020, worrlein_using_2024}. 

Our results demonstrate that it is possible to model accurately the process adn energy energy profiles form historical data.

The approach exploits the fact that energy profiles exhibit recurring execution patterns driven by the underlying process, the product, and external conditions. These patterns can be systematically extracted using Dynamic Time Warping (DTW) and subsequently reproduced with data-driven techniques. Our findings indicate that the most effective configuration combines DTW-based pattern extraction with ML regression over external factors; by contrast, sequence-to-sequence architectures yielded consistently lower performance in our setting \citep{worrlein_using_2024}.

The resulting profiles provide a more realistic representation of industrial process behaviour than methdos that only use punctual median values per activity \citep{gonzalez_process_2025}, and capture the temporal interplay between process execution, product characteristics, and the external factors that influence energy demand.

\vspace{\baselineskip}
\textbf{Process models, and the energy profiles derived from them, are strongly time-dependent}

Process modelling is frequently evaluated using control-flow metrics with little emphasis on activity and case durations. This omission is critical for energy prediction.

Evaluation~1 shows that the proposed method, which explicitly accounts for total case duration, achieves a substantially better fit for the overall process span. Within the same evaluation, the \texttt{ml\_local} variant delivers the strongest performance on individual activity durations, outperforming \texttt{ml\_global}. A plausible explanation is that a dedicated model per activity faces a simpler learning problem than a single model covering the entire process. Both variants improve markedly on the baseline that draws durations from the median of a stochastic Petri net.

Evaluation~3 confirms the downstream effect. On the aggregate metric, which implicitly reflects duration, the budget-based method performs best, and the predicted energy profiles correspondingly provide the closest match to the test data. Duration modelling therefore propagates directly into energy profile quality.

\vspace{\baselineskip}
\textbf{Coupling process and energy modelling yields a more faithful representation of the industrial system}

Evaluation~3 shows that the best overall results are obtained by combining the strongest process model with the strongest energy model. Methods that disregard the process and rely on the production schedule alone attain competitive aggregate metrics, but visual inspection reveals that the shape of the resulting profiles is severely distorted. The schedule-direct method, in particular, fails to reproduce the form of the profile at all.

Process modelling is thus essential for energy profile generation, as it imposes a causally plausible and mechanistically grounded constraint on the resulting curves. Because of stochastic variation in the simulation, the generated profiles are not identical to any specific historical trace; they are, however, plausible realisations for the facility under study. Approaches that ignore the process cannot offer this guarantee: their aggregate statistics may appear acceptable, but the underlying form is wrong.

\vspace{\baselineskip}
\textbf{Joint modelling improves understanding of the underlying system}

Industrial energy modelling is commonly carried out independently of the process control flow, or at a highly aggregated level. Both practices obscure how process executions causally generate specific energy profiles. The proposed framework instead models the two jointly and transparently.

We demonstrated this through the combined visualisation of process and energy, showing material flow, machine activities, and their corresponding energy profiles, which we used here to compare the real and simulated cases. The same representation is equally applicable to descriptive and diagnostic analytics, allowing historical process behaviour and its associated energy demand to be examined and, for instance, individual activities to be linked to the demand they generated.

Taken together, the joint visualisation of process and energy, the structure of the digital twin, and the results of its simulations deepen the understanding of how process execution and energy consumption interact. The digital twin supports granular analysis---identifying, for example, precisely which process, product, and external factors produced a given peak load at a given time. By representing the causal logic of energy demand explicitly, the framework supports informed decision-making.

\subsection{Contributions to the literature}

First, we contribute to the literature on digital twin development for manufacturing systems and their use for simualtion. We argue that the causal factors influencing process execution must be incorporated into the modelling procedure, and that this modelling should be carried out structurally: an energy profile cannot be represented accurately without the mechanistic, causally grounded process and product information that generates it.

Second, we contribute to the process mining literature on data-driven modelling of industrial processes. We argue and demonstrate that separating material flow from machine activities is essential for the granular modelling of energy profiles, and that temporal information---both at the level of individual activity executions and across total case durations---is equally central to this representation.

Third, we contribute to the literature on energy modelling of industrial processes, where modelling is predominantly based on aggregate prediction or on time series prediction using the current values of process variables. We extend this by modelling time series on the basis of discrete event data, and by contributing methods that improve the accuracy of the resulting representation.

\subsection{Practical implications}

For practitioners modelling industrial system operations, our findings argue against treating process and energy as separate modelling problems. The XXX framework offers a first blueprint for constructing coupled process-and-energy digital twins. As the evaluations show, process-aware modelling is the only family of approaches capable of simulating energy profiles that remain faithful to the underlying execution.

We recommend combining the strongest available process discovery technique with the budget-based approach to bounding simulation time; together these produce both realistic event sequences and plausible total case durations.

For activity-level duration prediction, we recommend the \texttt{ml\_local} method, that is, one model trained per activity, since the individual prediction task is easier to learn. The \texttt{ml\_global} alternative remains competitive and is considerably cheaper computationally, as it requires training only a single model per process.

Finally, for individual energy profile prediction, we recommend the combination of DTW, machine learning regression, and external factors affecting energy demand. This configuration achieves better predictive performance while being simpler and less training-intensive than sequence-to-sequence alternatives.


\subsection{Limitations and future work}
\label{Limitations and future work}

This work focuses on the process and energy modelling of production processes, with particular emphasis on material flow, machine activities, and activity durations. It demonstrates how these factors influence energy profiles and must be incorporated to generate accurate energy predictions. However, we do not consider all aspects that can be important for process modelling and simulation, such as the effect of work-in-process on the model or the waiting time before starting an activity  \citep{camargo_learning_2023}, as these were not relevant for the industrial processes analysed in this study. We focus on case-based modelling, leaving object-centric process mining and simulation for future work \citep{van_der_aalst_object-centric_2023, knopp_discovering_2023}. Extending the model with more advanced process-modelling techniques could enable finer-grained process and energy modelling.

We conducted a large-scale evaluation using both simulated and real energy profiles to empirically validate the methods, predicting the exact shape of highly volatile profiles remains challenging. Although our method outperforms aggregated metrics, the simulated curves can still deviate from real-world anomalies. Future work should isolate individual energy profiles to determine which variables are necessary and sufficient for accurate shape prediction, and identify operational thresholds where data-driven modelling becomes unreliable.

Also, we focused on prediion of the eenrgy profiles based on correlations, other approache i. the future could also try to modelt he idnviudal profiels causally, for itnance base don causal graphs oif the operaitons. For this is is also enecsary to have form the amchines the set valeus and optation aprameters.

This paper focuses on the extraction and simulation of Process and Energy Digital Twins, but the resulting dsfsadnaflks were not evaluated against specific optimization benchmarks (e.g., net energy savings or financial ROI). Future research could deploy these modelling principles in prescriptive use cases, such as the optimal sizing of thermal storage tanks, battery integration, or energy-aware production scheduling.





\section{Conclusions}
\label{Conclusions}

% Motivated by the need for more sustainable data centers and the limits of purely data-driven modelling of their operations, we proposed and evaluated the TempCaFe framework for selecting causal features based on causal discovery algorithms and human domain knowledge. Our experimental findings not only suggest that the identified causal feature sets improve prediction performance, especially in interventional settings, but also require less training data (both in terms of features and samples) and are easier to interpret due to their grounding in causal knowledge of the system under investigation. Overall, our study highlights the value of integrating causal inference and domain knowledge with flexible ML algorithms to improve prediction and decision-making in energy systems like data centers.

\section{Data availability}
The code is available in an online repository. Data from the simulation and physical testbed is openly accessible. Data from the operational data centers is confidential and not provided.

\section{Declaration of competing interest}
The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

\section{Declaration of generative AI and AI-assisted technologies in the manuscript preparation process.}

During the preparation of this work, the authors used ChatGPT in order to paraphrase. After using this tool, the authors reviewed and edited the content as needed and take full responsibility for the content of the published article.

\section{Acknowledgments}


\section{Data availability}
The data of the simualtion is online avalaible for complete reproducibility. The data of the real industrial processes is confidential and cannot be shared.






\appendix
\section{My Appendix}
Appendix sections are coded under \verb+\appendix+.



\printcredits


\bibliographystyle{cas-model2-names}

\bibliography{references}




\end{document}

