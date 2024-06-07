# How to contribute to ML-Conf

## Reporting an Issue
We use the Issue tracking feature to track bugs and new code contributions. 

### Creating a bug report
Each issue should include a title and a description of the error you are facing. Ensure to include as much relevant information as possible, including a code sample or failing test demonstrating the expected behavior and your system configuration. Your goal should be to make it easy for yourself - and others - to reproduce the bug and figure out a fix.

### Feature Requests
Since this is a growing package, we welcome new feature requests! However, remember that you may need to write the code for a feature yourself. Depending on the type of feature you want, there are slightly different requirements. Some examples are:
- **Requesting a utility for an ML Pipeline.**
If this is an easy fix and we feel this would be helpful to many users facing the same issue, we would love to work with you on this to make it happen! 
- **Adding a new risk or defense.** 
Are you a researcher who has discovered a new risk or way to defend against known risks? We welcome your contributions! However, in most cases, we only include state-of-the-art risks or defenses in our package. The package aims to allow other users to test their models against known risks or defenses or enable researchers to test their techniques against the current state-of-the-art. Thus, having a peer-reviewed paper to justify adding a new risk or defense would be nice.

## Contributing Code
For new utilities that help with an ML pipeline, please submit an issue, and we can work together to find the best way to incorporate the utility. 

For a module incorporating a new risk or defense, we highly recommend following the same coding conventions as the rest of the package. To contribute, please follow these steps:
1. Clone or fork the repository.
2. Create a new branch for the new module.
3. Add a new file with your risk or defense under `mlconf/risk` or `mlconf/defense`, respectively.
4. Create a class, we follow a convention for all our risks and defenses:
    - The model on which to apply the technique should be an input to the class. 
    - The `__init__()` method should have most of the parameters required to run the technique. This allows users to use functions like `__attr__()` to log the parameters while running the technique. Optional parameters may go in other methods.
    - Where possible, use the utils available in the package.
    - If your module uses hyperparameters, please **recommend a reasonable default value**. 
    - Add docstrings to the class and methods, including the recommended range for specific hyperparameters for your technique. Please use the modules we have written as a reference for formatting these. 
    - We follow the [convention of prefixing the method with an underscore](https://docs.python.org/2/tutorial/classes.html#private-variables-and-class-local-references) for private attributes and methods in the class.
5. **Please include the code to download any files required by the module.**
6. Add an example script in `mlconf/defenses` or `mlconf/risks` to show how to use your technique. Please take a look at the existing examples to get an idea of how to use it. 
7. Run PyLint on your code using the `pylintrc` file in the repo to validate the code style.
8. If the technique requires additional packages, include them in the `environment.yml` file with the appropriate version number.
9. Submit a pull request referencing the issue. 
