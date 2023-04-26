# INTeractive learning via REPresentatIon Discovery

Version: 0.0.1 beta

INTREPID (stands for INTeractive learning via REPresentatIon Discovery) is a library that contains various interactive learning algorithms that learn a representation (or a latent state) from observational data in order to complete their tasks. 

A list of algorithms, environments, and utils are given below. For full details see [Wiki](https://github.com/microsoft/Intrepid/wiki) of this repository.

## What is Interactive Learning and Representation Discovery

Consider any agent, also called decision maker, (e.g., a bot, robot, LLM) that is taking actions in an environment (e.g., a place, an OS). The world changes as a effect of the agent's action and also because of other noise in (e.g., a person maybe moving in the background or an OS may receive a notification unrelated to what the bot did). The goal of this agent is to solve a task, e.g., navigate safetly to a given location, or compose an email and send it off. The agent maybe take a series of actions to accomplish its goal. This is called an *Interactive Learning* task as the agent interacts with the world.

Typically, the observes the world in the form of a complex representation (e.g., an image, a large piece of text). It is useful for solving the task to map this representation into a more manageable representation from which it is easier to learn how to take actions, or model the dynamics of the world. A large-language-model, for example, does this internally by predicting the next token using a large corpus of data. However, for a robot or software, there can be other approaches for learning this representation. This is called the *Representation Discovery* problem.

As the name suggests, INTREPID is designed to discover representation for the purpose of interactive learning. INTREPID is designed to contain state-of-the-art algorithms for mapping the observation into representation for the purpose of taking actions, learning the world model, debugging, or visualization. INTREPID also contains various interactive learning algorithms that use this representation to optimize reward, and learn a policy. This is an continual evolving repository with a lot of features to come over time. To contribute, see the contributing section below.

## Algorithms Currently Supported

1. **Homer**. Learns latent state/representation using temporal constrastive learning loss. Provably explores and optimizes reward in Block MDP setting. 
          
   **Citation**: Kinematic State Abstraction and Provably Efficient Rich-Observation Reinforcement Learning, _Dipendra Misra, Mikael Henaff, Akshay Krishnamurthy, John Langford_ [\[ICML 2020\]](http://proceedings.mlr.press/v119/misra20a/misra20a.pdf)
        
2. **PPE: Path Prediction and Elimination**. Learns latent state/representation using a variant of multi-step inverse dynamics where the model predicts the identity of the path (sequence of actions) used to reach a given observation. Provably ignores noisy TV like temporal distractions. A very fast and scalable algorithm for near-deterministic problems.

   **Citation**: Provable RL with Exogenous Distractors via Multistep Inverse Dynamics, _Yonathan Efroni, Dipendra Misra, Akshay Krishnamurthy, Alekh Agarwal, and John Langford_ [\[ICLR 2022 Oral\]](https://openreview.net/pdf?id=RQLLzMCefQu)

3. **RicHID**: Algorithm designed for control problems where the latent dynamics are [LQR](https://en.wikipedia.org/wiki/Linear%E2%80%93quadratic_regulator) but the LQR state is obfuscated by rich observational noise. Provably explores and extracts the latent state by predicting past actions from observation.

    **Citation**: Learning the Linear Quadratic Regulator from Nonlinear Observations, _Zakaria Mhammedi, Dylan J. Foster, Max Simchowitz, Dipendra Misra, Wen Sun, Akshay Krishnamurthy, Alexander Rakhlin, John Langford_ [\[NeurIPS 2020\]](https://papers.nips.cc/paper/2020/file/a70145bf8b173e4496b554ce57969e24-Paper.pdf)

4. **FactoRL (pron. Factorel)**: Algorithm designed for settings where the latent representation is factorized over a set of states with sparse dynamical evolution (Factored MDP dynamics). The latent state is extracted by performing independence test over pairs of atoms (e.g., pixels/tokens) of the observation. This is followed by performing contrastive learning similar to in-painting. The algorithm has guarantee of success under certain assumptions. However, the algorithm is quite computationally expensive to run.

     **Citation**: Provable Rich Observation Reinforcement Learning with Combinatorial Latent States, _Dipendra Misra, Qinghua Liu, Chi Jin, John Langford_ [\[ICLR 2021\]](https://openreview.net/pdf?id=hx1IXFHAw7R) [\[RL Theory Seminar\]](https://www.youtube.com/watch?v=SEE5Snqhd40&ab_channel=RLtheoryseminars)

5. **Sabre**: Sabre is an algorithm for Safe Reinforcement Learning that assumes a safety function that can only provide binary feedback (safe/unsafe). This safety function is unknown but can be queried. The algorithm works by performing a sequence of active learning queries for safety while ensuring possible coverage so safety can be learned everywhere. Under certain assumptions, the algorithm can guarantee never taking any unsafe action even during training, optimizing calls to safety, and finding safest optimal policy.

    **Citation**: Provable Safe Reinforcement Learning with Binary Feedback, _Andrew Bennett, Dipendra Misra, and Nathan Kallus_ [\[AISTATS 2023\]](https://arxiv.org/pdf/2210.14492.pdf)
    
In addition to the above, there is also support for frequently used tabular RL methods (e.g., Q-learning with Bonus [\[Jin et al. 2018\]](https://arxiv.org/pdf/1807.03765.pdf)) and policy search methods (e.g., Fitted Q-Iteration [\[Nan Jiang's note on FQI\]](https://nanjiang.cs.illinois.edu/files/cs598/note5.pdf) and Policy Search by Dynamic Programming [\[Bagnell et al., 2003\]](https://papers.nips.cc/paper_files/paper/2003/hash/3837a451cd0abc5ce4069304c5442c87-Abstract.html)). 

We hope to include more algorithms in the future particularly those for representation discovery via self-supervised learning, and any RL algorithm that has provable regret or PAC guarantees which are typically unavailable in popular DeepRL repositories.

## Environments currently supported

1. Challenging Block MDP environments: This includes Diabolical Combination Lock [\[Misra et al., 2020\]](http://proceedings.mlr.press/v119/misra20a/misra20a.pdf)
2. Simple Newtonian Mechanics LQR problem
3. Wrappers for OpenAI Gym, Matterport Simulator, Minigrid, and AI2Thor. You will need to install these packages on your own. We provide no guarantee for these repositories. See Wiki for details. 

## Basic Usage in under 1 minute

The code is built primarily using Python with PyTorch and is regularly tested on OSX and Linux Systems. However, the code should also work on any other system which supports these base dependencies. To run a sample code in under a minute do the following.

1. Git clone the repository `git clone https://github.com/microsoft/Intrepid.git`

2. Go to the Intrepid folder in a terminal.

3. Install requirements. If you are using pip then you can install them as `python3 -m pip install requirements.txt`. 

3. Run a sample code as `sh local_runs/run_homer.sh`. This will run the Homer algorithm on a toy task and should generate results inside the `./results` folder.

For full functionality please see the [wiki](https://github.com/microsoft/Intrepid/wiki) of this repository.

## Citing this repository

If you use this repository in your research and find it helpful, then please cite the usage as:

``` 
@software{Intrepid,

title = "Intrepid: INTeractive REPresentatIon Discovery, a library for decision making algorithms that learn latent state representation",

authors = "Dipendra Misra, Rajan Chari, Alex Lamb, Anurag Koul, Byron Xu, Akshay Krishnamurthy, Dylan Foster, John Langford",

year = "2023"

}
```

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA. 
Please use [PEP](https://peps.python.org/pep-0008/) standards for python programming if you send us pull request.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
