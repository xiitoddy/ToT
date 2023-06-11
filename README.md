Figure 1: Schematic illustrating various approaches to problem solving with LLMs. Each rectangle
box represents a thought, which is a coherent language sequence that serves as an intermediate
step toward problem solving. See concrete examples of how thoughts are generated, evaluated, and
searched in Figures 2,4,6.
choices instead of just picking one, and (2) evaluates its current status and actively looks ahead or
backtracks to make more global decisions.
To design such a planning process, we return to the origins of artiﬁcial intelligence (and cognitive
science), drawing inspiration from the planning processes explored by Newell, Shaw, and Simon
starting in the 1950s [ 18, 19 ]. Newell and colleagues characterized problem solving [18 ] as search
through a combinatorial problem space, represented as a tree. We thus propose the Tree of Thoughts
(ToT) framework for general problem solving with language models. As Figure 1 illustrates, while
existing methods (detailed below) sample continuous language sequences for problem solving, ToT
actively maintains a tree of thoughts, where each thought is a coherent language sequence that serves
as an intermediate step toward problem solving (Table 1). Such a high-level semantic unit allows the
LM to self-evaluate the progress different intermediate thoughts make towards solving the problem
through a deliberate reasoning process that is also instantiated in language (Figures 2,4,6). This
implementation of search heuristics via LM self-evaluation and deliberation is novel, as previous
search heuristics are either programmed or learned. Finally, we combine this language-based
capability to generate and evaluate diverse thoughts with search algorithms, such as breadth-ﬁrst
search (BFS) or depth-ﬁrst search (DFS), which allow systematic exploration of the tree of thoughts
with lookahead and backtracking.
Empirically, we propose three new problems that challenge existing LM inference methods even with
the state-of-the-art language model, GPT-4 [ 20 ]: Game of 24, Creative Writing, and Crosswords
(Table 1). These tasks require deductive, mathematical, commonsense, lexical reasoning abilities,
and a way to incorporate systematic planning or search. We show ToT obtains superior results on
all three tasks by being general and ﬂexible enough to support different levels of thoughts, different
ways to generate and evaluate thoughts, and different search algorithms that adapt to the nature of
different problems. We also analyze how such choices affect model performances via systematic
ablations and discuss future directions to better train and use LMs.
2 Background
We ﬁrst formalize some existing methods that use large language models for problem-solving,
which our approach is inspired by and later compared with. We use pθ to denote a pre-trained LM
with parameters θ, and lowercase letters x, y, z, s, · · · to denote a language sequence, i.e. x =
(x[1], · · · , x[n]) where each x[i] is a token, so that pθ (x) = ∏n
i=1 pθ (x[i]|x[1...i]). We use uppercase
letters S, · · · to denote a collection of language sequences.
Input-output (IO) prompting is the most common way to turn a problem input x into output y with
LM: y ∼ pθ (y|promptIO(x)), where promptIO(x) wraps input x with task instructions and/or few-
shot input-output examples. For simplicity, let us denote pprompt
θ (output | input) = pθ (output |
prompt(input)), so that IO prompting can be formulated as y ∼ pIO
θ (y|x).
2
Chain-of-thought (CoT) prompting [35 ] was proposed to address cases where the mapping of
input x to output y is non-trivial (e.g. when x is a math question and y is the ﬁnal numerical answer).
The key idea is to introduce a chain of thoughts z1, · · · , zn to bridge x and y, where each zi is a
coherent language sequence that serves as a meaningful intermediate step toward problem solving
(e.g. zi could be an intermediate equation for math QA). To solve problems with CoT, each thought
zi ∼ pCoT
θ (zi | x, z1···i−1) is sampled sequentially, then the output y ∼ pCoT
θ (y|x, z1···n). In
practice, [z1···n, y] ∼ pCoT
θ (z1···n, y|x) is sampled as a continuous language sequence, and the
decomposition of thoughts (e.g. is each zi a phrase, a sentence, or a paragraph) is left ambiguous.
Self-consistency with CoT (CoT-SC) [33 ] is an ensemble approach that samples k i.i.d. chains
of thought: [z(i)
1···n, y(i)] ∼ pCoT
θ (z1···n, y|x) (i = 1 · · · k), then returns the most frequent output:
arg maxy #{i | y(i) = y}. CoT-SC improves upon CoT, because there are generally different
thought processes for the same problem (e.g. different ways to prove the same theorem), and the
output decision can be more faithful by exploring a richer set of thoughts. However, within each
chain there is no local exploration of different thought steps, and the “most frequent” heuristic only
applies when the output space is limited (e.g. multi-choice QA).
3 Tree of Thoughts: Deliberate Problem Solving with LM
A genuine problem-solving process involves the repeated use of available informa-
tion to initiate exploration, which discloses, in turn, more information until a way
to attain the solution is ﬁnally discovered.—— Newell et al. [18]
Research on human problem-solving suggests that people search through a combinatorial problem-
space – a tree where the nodes represent partial solutions, and the branches correspond to operators
that modify them [ 18 , 19]. Which branch to take is determined by heuristics that help to navigate the
problem-space and guide the problem-solver towards a solution. This perspective highlights two key
shortcomings of existing approaches that use LMs to solve general problems: 1) Locally, they do not
explore different continuations within a thought process – the branches of the tree. 2) Globally, they
do not incorporate any type of planning, lookahead, or backtracking to help evaluate these different
options – the kind of heuristic-guided search that seems characteristic of human problem-solving.
To address these shortcomings, we introduce Tree of Thoughts (ToT), a paradigm that allows LMs to
explore multiple reasoning paths over thoughts (Figure 1(c)). ToT frames any problem as a search
over a tree, where each node is a state s = [x, z1···i] representing a partial solution with the input and
the sequence of thoughts so far. A speciﬁc instantiation of ToT involves answering four questions:
1. How to decompose the intermediate process into thought steps; 2. How to generate potential
thoughts from each state; 3. How to heuristically evaluate states; 4. What search algorithm to use.
1. Thought decomposition. While CoT samples thoughts coherently without explicit decomposition,
ToT leverages problem properties to design and decompose intermediate thought steps. As Table 1
shows, depending on different problems, a thought could be a couple of words (Crosswords), a line of
equation (Game of 24), or a whole paragraph of writing plan (Creative Writing). In general, a thought
should be “small” enough so that LMs can generate promising and diverse samples (e.g. generating
a whole book is usually too “big” to be coherent), yet “big” enough so that LMs can evaluate its
prospect toward problem solving (e.g. generating one token is usually too “small” to evaluate).
2. Thought generator G(pθ , s, k). Given a tree state s = [x, z1···i], we consider two strategies to
generate k candidates for the next thought step:
(a) Sample i.i.d. thoughts from a CoT prompt (Creative Writing, Figure 4): z(j) ∼
pCoT
θ (zi+1|s) = pCoT
θ (zi+1|x, z1···i) (j = 1 · · · k). This works better when the thought
space is rich (e.g. each thought is a paragraph), and i.i.d. samples lead to diversity;
(b) Propose thoughts sequentially using a “propose prompt” (Game of 24, Figure 2; Crosswords,
Figure 6): [z(1), · · · , z(k)] ∼ ppropose
θ (z(1···k)
i+1 | s). This works better when the thought
space is more constrained (e.g. each thought is just a word or a line), so proposing different
thoughts in the same context avoids duplication.
3. State evaluator V (pθ , S). Given a frontier of different states, the state evaluator evaluates the
progress they make towards solving the problem, serving as a heuristic for the search algorithm
to determine which states to keep exploring and in which order. While heuristics are a standard
approach to solving search problems, they are typically either programmed (e.g. DeepBlue [ 3]) or
3
learned (e.g. AlphaGo [ 26]). We propose a third alternative, by using the LM to deliberately reason
about states. When applicable, such a deliberate heuristic can be more ﬂexible than programmed
rules, and more sample-efﬁcient than learned models. Similar to the thought generator, we consider
two strategies to evaluate states either independently or together:
(a) Value each state independently: V (pθ , S)(s) ∼ pvalue
θ (v|s) ∀s ∈ S, where a value
prompt reasons about the state s to generate a scalar value v (e.g. 1-10) or a classiﬁca-
tion (e.g. sure/likely/impossible) that could be heuristically turned into a value. The basis
of such evaluative reasoning can vary across problems and thought steps. In this work, we
explore evaluation via few lookahead simulations (e.g. quickly conﬁrm that 5, 5, 14 can
reach 24 via 5 + 5 + 14, or “hot l” can mean “inn” via ﬁlling “e” in “ ”) plus commonsense
(e.g. 1 2 3 are too small to reach 24, or no word can start with “tzxc”). While the former
might promote “good” states, the latter could help eliminate “bad” states. Such valuations
do not need to be perfect, and only need to be approximately
(b) Vote across states: V (pθ , S)(s) = 1[s = s∗], where a “good” state s∗ ∼ pvote
θ (s∗|S) is
voted out based on deliberately comparing different states in S in a vote prompt. When
problem success is harder to directly value (e.g. passage coherency), it is natural to to instead
compare different partial solutions and vote for the most promising one. This is similar
in spirit to a “step-wise” self-consistency strategy, i.e. cast “which state to explore” as a
multi-choice QA, and use LM samples to vote for it.
For both strategies, we could prompt the LM multiple times to aggregate the value or vote results to
trade time/resource/cost for more faithful/robust heuristics.
Algorithm 1 ToT-BFS(x, pθ , G, k, V, T, b)
Require: Input x, LM pθ , thought generator G()
& size limit k, states evaluator V (), step limit T ,
breadth limit b.
S0 ← {x}
for t = 1, · · · , T do
S′
t ← {[s, z] | s ∈ St−1, zt ∈ G(pθ , s, k)}
Vt ← V (pθ , S′
t)
St ← arg maxS⊂S′
t,|S|=b
∑
s∈S Vt(s)
end for
return G(pθ , arg maxs∈ST VT (s), 1)
Algorithm 2 ToT-DFS(s, t, pθ , G, k, V, T, vth)
Require: Current state s, step t, LM pθ , thought
generator G() and size limit k, states evaluator
V (), step limit T , threshold vth
if t > T then record output G(pθ , s, 1)
end if
for s′ ∈ G(pθ , s, k) do . sorted candidates
if V (pθ , {s′})(s) > vthres then . pruning
DFS(s′, t + 1)
end if
end for
4. Search algorithm. Finally, within the ToT framework, one can plug and play different search
algorithms depending on the tree structure. We explore two relatively simple search algorithms and
leave more advanced ones (e.g. A* [9], MCTS [2]) for future work:
(a) Breadth-ﬁrst search (BFS) (Algorithm 1) maintains a set of the b most promising states
per step. This is used for Game of 24 and Creative Writing where the tree depth is limit
(T ≤ 3), and initial thought steps can be evaluated and pruned to a small set (b ≤ 5).
(b) Depth-ﬁrst search (DFS) (Algorithm 2) explores the most promising state ﬁrst, until the
ﬁnal output is reached (t > T ), or the state evaluator deems it impossible to solve the
problem from the current s (V (pθ , {s})(s) ≤ vth for a value threshold vth). In the latter
case, the subtree from s is pruned to trade exploration for exploitation. In both cases, DFS
backtracks to the parent state of s to continue exploration.
Conceptually, ToT has several beneﬁts as a method for general problem-solving with LMs: (1) Gener-
ality. IO, CoT, CoT-SC, and self-reﬁnement can be seen as special cases of ToT (i.e. trees of limited
depth and breadth; Figure 1). (2) Modularity. The base LM, as well as the thought decomposition,
generation, evaluation, and search procedures can all be varied independently. (3) Adaptability.
Different problem properties, LM capabilities, and resource constraints can be accommodated. (4)
Convenience. No extra training is needed, just a pre-trained LM is sufﬁcient. The next section will
show how these conceptual beneﬁts translate to strong empirical performance in different problems.
4 Experiments
We propose three tasks that are hard even when sampling from the state-of-the-art language model,
GPT-4 [ 20 ], using standard IO prompting or chain-of-thought (CoT) prompting. We show how
4
Game of 24 Creative Writing 5x5 Crosswords
Input 4 numbers (4 9 10 13) 4 random sentences 10 clues (h1. presented;..)
Output An equation to reach 24
(13-9)*(10-4)=24
A passage of 4 paragraphs
ending in the 4 sentences
5x5 letters: SHOWN;
WIRRA; AVAIL; ...
Thoughts 3 intermediate equations
(13-9=4 (left 4,4,10); 10-
4=6 (left 4,6); 4*6=24)
A short writing plan
(1. Introduce a book that
connects...)
Words to ﬁll in for clues:
(h1. shown; v5. naled; ...)
#ToT steps 3 1 5-10 (variable)
Table 1: Task overview. Input, output, thought examples are in blue.
deliberate search in trees of thoughts (ToT) produces better results, and more importantly, interesting
and promising new ways to use language models to solve problems requiring search or planning.
Unless otherwise stated, we perform experiments using a Chat Completion mode GPT-41 with a
sampling temperature of 0.7.
4.1 Game of 24
Game of 24 is a mathematical reasoning challenge, where the goal is to use 4 numbers and basic
arithmetic operations (+-*/) to obtain 24. For example, given input “4 9 10 13”, a solution output
could be “(10 - 4) * (13 - 9) = 24”.

Figure 2: ToT in a game of 24. The LM is prompted for (a) thought generation and (b) valuation.
Task Setup. We scrape data from 4nums.com, which has 1,362 games that are sorted from easy to
hard by human solving time, and use a subset of relatively hard games indexed 901-1,000 for testing.
For each task, we consider the output as success if it is a valid equation that equals 24 and uses the
input numbers each exactly once. We report the success rate across 100 games as the metric.
Baselines. We use a standard input-output (IO) prompt with 5 in-context examples. For chain-of-
thought (CoT) prompting, we augment each input-output pair with 3 intermediate equations, each
operating on two remaining numbers. For example, given input “4 9 10 13”, the thoughts could be
“13 - 9 = 4 (left: 4 4 10); 10 - 4 = 6 (left: 4 6); 4 * 6 = 24 (left: 24)”. For each game, we sample IO
and CoT prompting for 100 times for average performance. We also consider a CoT self-consistency
baseline, which takes the majority output from 100 CoT samples, and an iterative-reﬁne approach on
top of an IO sample for at most 10 iterations. At each iteration, the LM is conditioned on all previous
history to “reﬂect on your mistakes and generate a reﬁned answer” if the output is incorrect. Note
that it uses groundtruth feedback signals about equation correctness.
ToT Setup. To frame Game of 24 into ToT, it is natural to decompose the thoughts into 3 steps,
each an intermediate equation. As shown in Figure 2(a), at each tree node, we exact the “left”
numbers and prompt the LM to propose some possible next steps. The same “propose prompt” is
used for all 3 thought steps, though it only has one example with 4 input numbers. We perform a
breadth-ﬁrst search (BFS) in ToT, where at each step we keep the best b = 5 candidates. To perform
deliberate BFS in ToT, as shown in Figure 2(b), we prompt LM to evaluate each thought candidate as
“sure/maybe/impossible” with regard to reaching 24. The aim is to promote correct partial solutions
that can be verdicted within few lookahead trials, and eliminate impossible partial solutions based on
“too big/small” commonsense, and keep the rest “maybe”. We sample values 3 times for each thought.
1Experiments were done between May 5-16, 2023.
5
Method Success
IO prompt 7.3%
CoT prompt 4.0%
CoT-SC (k=100) 9.0%
ToT (ours) (b=1) 45%
ToT (ours) (b=5) 74%
IO + Reﬁne (k=10) 27%
IO (best of 100) 33%
CoT (best of 100) 49%
Table 2: Game of 24 Results.0 25 50 75 100
0.2
0.4
0.6
(a) Success rate with nodes visited
IO (best of k)
CoT (best of k)
ToT (b=1...5)1 2 3 4 Correct
0.0
0.2
0.4
0.6
(b) Samples failed at each step
CoT
ToT (b=5) Figure 3: Game of 24 (a) scale analysis & (b) error analysis.
Results. As shown in Table 2, IO, CoT, and CoT-SC prompting methods perform badly on the task,
achieving only 7.3%, 4.0%, and 9.0% success rates. In contrast, ToT with a breadth of b = 1 already
achieves a success rate of 45%, while b = 5 achieves 74%. We also consider an oracle setup for
IO/CoT, by calculating the success rate using best of k samples (1 ≤ k ≤ 100). To compare IO/CoT
(best of k) with ToT, we consider calculating the tree nodes visited per task in ToT across b = 1 · · · 5,
and map the 5 success rates in Figure 3(a), treating IO/CoT (best of k) as visiting k nodes in a bandit.
Not surprisingly, CoT scales better than IO, and best of 100 CoT samples achieve a success rate of
49%, but still much worse than exploring more nodes in ToT (b > 1).
Error Analysis. Figure 3(b) breaks down at which step CoT and ToT samples fail the task, i.e. the
thought (in CoT) or all b thoughts (in ToT) are invalid or impossible to reach 24. Notably, around
60% of CoT samples already failed the task after generating the ﬁrst step, or equivalently, the ﬁrst
three words (e.g. “4 + 9”). This highlights the issues with direct left-to-right decoding.
4.2 Creative writing
Next, we invent a creative writing task where the input is 4 random sentences and the output should
be a coherent passage with 4 paragraphs that end in the 4 input sentences respectively. Such a task is
open-ended and exploratory, and challenges creative thinking as well as high-level planning.
Task setup. We sample random sentences from randomwordgenerator.com to form 100 inputs, and
there is no groundtruth passage for each input constraint. As we ﬁnd that GPT-4 can follow the
input constraints most of the time, we focus on evaluating passage coherency in two ways: using a
GPT-4 zero-shot prompt to provide a 1-10 scalar score, or using human judgments to compare pairs
of outputs from different methods. For the former, we sample 5 scores and average them for each task
output, and we ﬁnd these 5 scores usually consistent, with a standard deviation of around 0.56 on
average across outputs. For the latter, we employ a subset of the authors in a blind study to compare
the coherency of CoT vs. ToT generated passage pairs, where the order of passages is random ﬂipped
over 100 inputs.
Baselines. Given the creative nature of the task, both IO and CoT prompts are zero-shot. While the
former prompts the LM to directly generate a coherent passage given input constraints, the latter
prompts the LM to ﬁrst make a brief plan then write the passage, i.e. the plan serves as the intermediate
thought step. We generate 10 IO and CoT samples per task. We also consider an iterative-reﬁne
(k ≤ 5) method on top of a random IO sample for each task, where the LM is conditioned on input
constraints and the last generated passage to decide if the passage is already “perfectly coherent”,
and if not generate a reﬁned one.
ToT setup. We build a ToT with depth 2 (and only 1 intermediate thought step) — the LM ﬁrst
generates k = 5 plans and votes for the best one (Figure 4), then similarly generate k = 5 passages
based on the best plan then vote for the best one. Here the breadth limit b = 1, as only one choice is
kept per step. A simple zero-shot vote prompt (“analyze choices below, then conclude which is most
promising for the instruction”) is used to sample 5 votes at both steps.
Results. Figure 5(a) shows average GPT-4 scores across 100 tasks, where ToT (7.56) is deemed to
generate more coherent passages than IO (6.19) and CoT (6.93) on average. While such an automatic
metric might be noisy, Figure 5(b) conﬁrms the ﬁnding by showing that humans prefer ToT over
CoT in 41 out of 100 passage pairs, while only prefer CoT over ToT in 21 (other 38 pairs are found
“similarly coherent”). Lastly, iterative-reﬁne is more effective on this natural language task, where
6
 4: A step of deliberate search in a randomly picked Creative Writing task. Given the input, the
LM samples 5 different plans, then votes 5 times to decide which plan is best. The majority choice is
used to consequently write the output passage with the same sample-vote procedure.IO CoT ToT IO
+refine
ToT
+refine
4
6
8
(a) GPT-4 coherency scoresCoT > ToT Similar ToT > CoT
0
10
20
30
40
21
38 41
(b) Human coherency comparison
Figure 5: Creative Writing results.
Method Success Rate (%)
Letter Word Game
IO 38.7 14 0
CoT 40.6 15.6 1
ToT (ours) 78 60 20
+best state 82.4 67.5 35
-prune 65.4 41.5 5
-backtrack 54.6 20 5
Table 3: Mini Crosswords results.
it improves IO coherency score from 6.19 to 7.67, and ToT coherency score from 7.56 to 7.91. We
believe it could be thought of as a third approach to thought generation in the ToT framework, where
new thoughts can arise from reﬁning old thoughts instead of i.i.d. or sequentially generated.
4.3 Mini Crosswords
In Game of 24 and Creative Writing, ToT is relatively shallow — at most 3 thought steps are needed
to reach the ﬁnal output. Here we explore 5 × 5 mini crosswords as a harder search problem involving
natural language. Again, the goal is not just to solve the task, as more general crosswords can be
readily solved with specialized NLP pipelines [31] that leverages large-scale retrieval instead of LM.
Rather, we aim to explore the limit of LM as a general problem solver that explores its own thoughts
and guides its own exploration with deliberate reasoning as heuristics.
Task Setup. We scrape data from GooBix, which contains 156 games of 5 × 5 mini crosswords. As
we observe adjacent games contain similar clues, we use 20 games with indices 1, 6, · · · , 91, 96 for
testing, and games 136, 141, 146, 151, 156 for prompting. For each task, the input describes the 5
horizontal clues and 5 vertical clues, and the output should be a board of 5 × 5 = 25 letters to solve
the crosswords. For evaluation, we consider three levels of success: the portion of correct letters (25
per game), words (10 per game), and games.
Baselines. We provide 5 example input-output pairs in the IO prompt, and in the CoT prompt
additionally include intermediate words in the order h1..5 then v1..5. We run each prompt for 10
samples and average the results.
ToT Setup. We leverage a depth-ﬁrst search (Algorithm 2) that keeps exploring the most promising
subsequent word clue until the state is no longer promising, then backtrack to the parent state to
explore alternative thoughts. To make search tractable, subsequent thoughts are constrained not to
change any ﬁlled words or letters, so that the ToT has at most 10 intermediate steps. For thought
generation, at each state we translate all existing thoughts (e.g. “h2.motor; h1.tasks” for the state
in Figure 6(a)) into letter constraints for remaining clues (e.g. “v1.To heap: tm ;...”) and prompt
a proposal prompt 5 times to come up with candidates for where and what to ﬁll in the next word.
Importantly, we also prompt the LM to give a conﬁdence level for different thoughts, and aggregate
7
Figure 6: In Mini Crosswords, (a) how thoughts are proposed and aggregated in a priority queue
for depth-ﬁrst search (DFS), and (b) how a state is evaluated based on the possibility of ﬁlling in
each remaining word clue, and pruned if any remaining clue is deemed not possible to ﬁll by the LM.
Then DFS backtracks to the parent state and explore the next promising thought for clue.
these across proposals to obtain a sorted list of next thoughts to explore (Figure 6(a)). For state
evaluations, we similarly translate each state into letter constraints for remaining clues, then evaluate
for each clue if it is possible to ﬁll given the constraints. If any remaining clue is deemed “impossible”
to ﬁll in (e.g. “v1. To heap: tm s ”), then the exploration of the state’s subtree is pruned and DFS
backtracks to its parent to explore the next promising thought. We limit DFS search steps to 100, and
simply render the deepest explored state (the ﬁrst explored one if multiple) into the ﬁnal output.
Results. As shown in Table 3, IO and CoT prompting methods perform poorly with a word-level
success rate less than 16%, while ToT signiﬁcantly improves all metrics, achieving a word-level
success rate of 60% and solving 4 out of 20 games. Such an improvement is not surprising, given IO
and CoT lack mechanisms to try different clues, make changes to decisions, or backtrack.
Oracle and ablation studies. When outputting from the oracle best DFS state (instead of the
heuristically determined best state) per task, ToT performance is even higher and actually solves
7/20 games (Table 3, “+best state”), indicating our simple output heuristics can be readily improved.
Interestingly, sometimes when the crosswords game is actually solved, the state evaluator might still
deem some words as “impossible” and prune — possibly because 5 × 5 crosswords by design have
some rare or obselete words that GPT-4 cannot recognize2. Given the state evaluation as a pruning
heuristic is imperfect, we also explore ablating the pruning, and ﬁnd the performance generally worse
(Table 3, “-prune”). However, it could actually ﬁnd the correct solution for 4/20 games (though only
outputting 1 via heuristic), 3 of which are games ToT+pruning cannot solve within 100 steps. Thus,
better heuristics for DFS pruning are critical for problem solving in this case. Lastly, we conﬁrm the
importance of backtracking by running an ablation that keeps ﬁlling the most promising clue for at
most 20 steps, allowing overwrites. This is similar to a “greedy” BFS search with breadth limit of
b = 1, and performs poorly with a word level success of only 20% (Table 3, “-backtrack”).
5 Related Work
Planning and decision making. Smart planning and decision making are critical to achieving
predeﬁned goals. As they are trained on vast amount of world knowledge and human examples, LMs
are known to have already absorbed rich commonsense that makes it possible to propose reasonable
plans conditioned on problem setting and environmental states [ 10 , 39 , 34, 11 , 32 , 38, 37]. Our
proposed Tree-of-Thought approach extends existing planning formulations by considering multiple
potentially feasible plans simultaneously at each problem-solving step, and proceeding with the most
promising ones. The integration between thought sampling and value feedback organically integrates
planning and decision-making mechanisms, enabling effective search inside a solution tree. On the
other hand, traditional decision-making procedures usually require training dedicated reward and
policy models as in reinforcement learning (for example CHAI [ 30]), whereas we use the LM itself
to provide the value estimates for decision making.
2For example, “agend” is an obsolete form of “agendum”, but GPT-4 deems it a typo for “agenda”. External
retrieval or web interaction could augment LM for problem solving under knowledge uncertainty.
8
Self-reﬂection. Using LLMs to assess the viability of their own predictions is becoming an in-
creasingly important procedure in problem solving. [25 , 17 , 21 ] introduced the “self-reﬂection”
mechanism, in which LMs provide feedback to their generation candidates. [ 4 ] improves LMs code
generation accuracy by injecting feedback messages generated by the LM itself based on its code
execution results. Similarly, [ 14 ] also introduces “critic” or review steps over the actions and states,
deciding the next action to take in solving computer operation tasks. Another recent work very
relevant to ours is “self-eval guided decoding” [ 36]. Similar to our method, self-eval decoding
also follows a tree-search procedure with leaves sampled from stochastic beam search decoding,
which are then evaluated by LLM itself with carefully prepared self-eval prompts. Their approach
however, uses the PAL formulation [7] which represents thoughts as codes, which makes it difﬁcult
to tackle challenging tasks like creative writing which we consider in this paper. Our Tree-of-Thought
formulation is thus more versatile and handles challenging tasks on which GPT-4 only achieves very
low accuracy with standard prompts.
Program-guided LLM generation. Our proposal is also related to recent advancements that or-
ganize LM’s behavior with symbolic program guidance. For example [ 24] embeds LMs in an
algorithmic search procedure to help solve problems like question answering step-by-step, in which
the search trees are expanded by relevant paragraphs that might provide answers. This approach
however differs from ours in that trees are expanded by sampling external paragraphs instead of the
LM’s own thoughts, and there is no reﬂection or voting steps. Another approach, LLM+P [ 15], goes
one step further and delegates the actual planning process to a classical planner.
Classical search methods. Last but not least, our approach can be treated as a modern rendition
of classical search methods for problem solving. For example it can be considered as a heuristic
search algorithm like A* [8 ], in which the heuristic at each search node is provided by the LM’s
self-assessment. From this perspective, our method is also related to NeuroLogic A*esque decoding
proposed in [ 16 ], which is inspired by A* search but introduces look-ahead heuristics that are
efﬁcient for LMs to improve the beam-search or top-k sampling decoding. This method however is
constrained to sentence generation tasks, whereas our framework are designed for complex, multi-step
problem solving guarded by value feedback.
6 Discussion
Limitations and future directions. Deliberate search such as ToT might not be necessary for
many existing tasks that GPT-4 already excels at, and as an initial step this work only explores
three relatively simple tasks that challenges GPT-4 and calls of better search and planning abilities
incorporated with LMs. However, as we begin to deploy LMs for more real-world decision making
applications (e.g. coding, data analysis, robotics, etc.), more complex tasks could emerge and present
new opportunities to study these research questions. Also, search methods like ToT requires more
resources (e.g. GPT-4 API cost) than sampling methods in order to improve task performances,
but the modular ﬂexibility of ToT allows users to customize such performance-cost tradeoffs, and
ongoing open-source efforts [ 29 ] should readily reduce such costs in the near future. Lastly, this work
focuses on using an off-the-shelf LM, and ﬁne-tuning LMs using a ToT-style high-level counterfactual
decision making (e.g. deliberating over potential choices for the next paragraph, instead of predicting
the next token) might present opportunities to enhance the problem-solving capabilities of LMs.
Broader impact. ToT is a framework that empowers LMs to more autonomously and intelligently
make decisions and solve problems. While current tasks are limited to reasoning and search problems,
future applications involving interaction with external environments or humans could bring potential
danger, e.g. facilitating harmful uses of LMs. On the other hand, ToT also improves the interpretability
of model decisions and the opportunity for human alignment, as the resulting representations are
readable, high-level language reasoning instead of implicit, low-level token values.
Conclusion. The associative “System 1” of LMs can be beneﬁcially augmented by a “System 2”
based on searching a tree of possible paths to the solution to a problem. The Tree of Thoughts
framework provides a way to translate classical insights about problem-solving into actionable
methods for contemporary LMs. At the same time, LMs address a weakness of these classical
methods, providing a way to solve complex problems that are not easily formalized, such as creative
writing. We see this intersection of LMs with classical approaches to AI as an exciting direction for
future work.
