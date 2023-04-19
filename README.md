# Intrepid: INTeractive REPresentatIon Discovery

Version: 0.0.1 beta

This repository contains various decision making algorithms that learn a representation (or a latent state) from observational data
in order to complete their tasks. 

A list of algorithms, environments, and utils are given below. For full details see Wiki of this repository.

## Algorithms Currently Supported

1. **Homer**. Learns latent state/representation using temporal constrastive learning loss. Provably explores and optimizes reward in Block MDP setting. 
          
   **Citation**: Kinematic State Abstraction and Provably Efficient Rich-Observation Reinforcement Learning, _Dipendra Misra, Mikael Henaff, Akshay Krishnamurthy, John Langford_ [\[ICML 2020\]](http://proceedings.mlr.press/v119/misra20a/misra20a.pdf)
        
2. **PPE: Path Prediction and Elimination**. Learns latent state/representation using a variant of multi-step inverse dynamics where the model predicts the identity of the path (sequence of actions) used to reach a given observation. Provably ignores noisy TV like temporal distractions. A very fast and scalable algorithm for near-deterministic problems.

   **Citation**: Provable RL with Exogenous Distractors via Multistep Inverse Dynamics, _Yonathan Efroni, Dipendra Misra, Akshay Krishnamurthy, Alekh Agarwal, and John Langford_ [\[ICLR 2022 Oral\]](https://openreview.net/pdf?id=RQLLzMCefQu)

3. **RicHID**: Algorithm designed for control problems where the latent dynamics are [LQR](https://en.wikipedia.org/wiki/Linear%E2%80%93quadratic_regulator) but the LQR state is obfuscated by rich observational noise. Provably explores and extracts the latent state by predicting past actions from observation.

    **Citation**: Learning the Linear Quadratic Regulator from Nonlinear Observations, _Zakaria Mhammedi, Dylan J. Foster, Max Simchowitz, Dipendra Misra, Wen Sun, Akshay Krishnamurthy, Alexander Rakhlin, John Langford_ [\[NeurIPS 2020\]](https://papers.nips.cc/paper/2020/file/a70145bf8b173e4496b554ce57969e24-Paper.pdf)

4. **FactoRL**:

5. PSDP: Policy Search by Dynamic Programming

6. FQI: Fitted Q-Iteration

7. 

## Environments currently supported

1. Challenging Block MDP environments: This includes Diabolical Combination Lock [Misra et al., 2020](http://proceedings.mlr.press/v119/misra20a/misra20a.pdf)
2. Simple Newtonian Mechanics LQR problem
3. Wrappers for OpenAI Gym, Matterport Simulator, Minigrid, and AI2Thor. You will need to install these packages on your own. We provide no guarantee for these repositories. See Wiki for details. 

## Basic Usage in under 1 minute

1. Git clone the repository.

2. Go to the experiment folder.

3. Run a sample code as `sh local_runs/run_homer.sh`

## Citing this repository

If you use this repository in your research and find it helpful, then please cite the usage as:

``` 
{Intrepid,

title="Intrepid: INTeractive REPresentatIon Discovery, a library for decision making algorithms that learn latent state representation",

authors="Dipendra Misra, Rajan Chari, Alex Lamb, Anurag Koul, Akshay Krishnamurthy, Dylan Foster, John Langford",

year="2023"

}
```


#### To remove stuff below ####

# Project

> This repo has been populated by an initial template to help get you started. Please
> make sure to update the content to build a great experience for community-building.

As the maintainer of this project, please make a few updates:

- Improving this README.MD file to provide a great experience
- Updating SUPPORT.MD with content about this project's support experience
- Understanding the security reporting process in SECURITY.MD
- Remove this section from the README

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
