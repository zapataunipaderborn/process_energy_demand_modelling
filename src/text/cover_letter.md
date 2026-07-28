\documentclass[preprint,12pt]{elsarticle}

%% The amssymb package provides various useful mathematical symbols
\usepackage{amssymb}
\usepackage{url}
%% The amsmath package provides various useful equation environments.
\usepackage{amsmath}
%% Compact one-page layout: wide text block, small margins, no top space
\usepackage[a4paper,margin=1.5cm,top=1cm,bottom=1cm]{geometry}

%\journal{Computers & Industrial Engineering}

\begin{document}

\begin{center}
{\Large\textbf{Cover Letter}}
\end{center}

\noindent Guest Editors of Computers \& Industrial Engineering\\
Special issue: \textit{Simulation-Driven Industrial Transformation: Towards Resilient, Sustainable, and Human-Centric Operations}

\vspace{\baselineskip}
\noindent Dear Guest Editors,

\vspace{\baselineskip}
\noindent On behalf of my co-authors, I am pleased to submit our manuscript entitled ``Process and Energy Digital Twins: Modelling Industrial Processes and Energy Profiles with Process Mining and Machine Learning'' for consideration in the Computers \& Industrial Engineering special issue \textit{Simulation-Driven Industrial Transformation: Towards Resilient, Sustainable, and Human-Centric Operations}.

Our study investigates the data-driven modelling and simulation of industrial processes together with the energy profiles they generate. Concretely, we propose a framework that extracts a Process and Energy Digital Twin of a production facility directly from historical data: Process Mining extracts a granular process model in which time is modelled explicitly, and Machine Learning extracts continuous energy profile models per machine activity, conditioned on product attributes and external factors such as the weather. Coupled, the models simulate a production plan into complete energy profiles. We evaluate the framework in three evaluations---of the process, of the individual energy profiles, and of the complete profiles of simulated cases---on six industrial processes: one synthetic process with known ground truth and five processes from three real food-production facilities, together comprising 766 cases and 79 energy time series over close to 7{,}000 hours of operation.

The results demonstrate that only process-aware simulation reproduces realistic energy profiles: approaches that skip the process may match aggregate statistics, but fail to reproduce the shape of the real energy behaviour. The machine learning duration models outperform the commonly used stochastic distributions, and the proposed energy profile models predict more realistic curves than sequence-to-sequence alternatives and median-based baselines. The resulting digital twin links every simulated profile to its causes---process execution, durations, product characteristics, and external factors---supporting energy-aware planning and decision support in industry.

This paper aligns closely with the themes of the special issue, in particular ``Digital Twins and Real-Time Simulations'', ``AI-Powered Simulations for Decision Support and Optimization'', ``Energy Optimization through Simulation'', ``Complex Industrial Systems Simulation'', and ``Simulation-Driven Decision Support Systems''. By enabling industries to test production scenarios and evaluate their energy impact in a virtual environment before implementation, our framework directly supports the transition toward resilient, sustainable, and energy-efficient industrial operations envisioned by Industry 4.0 and Industry 5.0. It also contributes to the UN Sustainable Development Goal of affordable and clean energy, particularly in the pursuit of energy-efficient and climate-neutral manufacturing.

We confirm that this manuscript has not been published elsewhere and is not under consideration for publication in another journal.

Thank you for considering our submission. We believe that our study will be of strong interest to the readership of Computers \& Industrial Engineering and the special issue, and we look forward to your feedback.

\vspace{\baselineskip}
\noindent Sincerely,

\vspace{\baselineskip}
\noindent David Zapata Gonzalez\\
On behalf of all co-authors

\end{document}
