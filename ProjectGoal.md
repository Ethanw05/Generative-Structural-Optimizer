An app that allows the user to
1. Import a Part or Assembly from solid works
2. Define Fixtures, Forces(Crash tests, motion studies, vibrations, torque, regular static forces, oscillations, and anything else a mechanism could do), exclusions, and other conditions around a system in an interactive 3D view port
3. Run Generative Optimization on it which shows the part after optimization. (If the optimization is evolutionary you should be able to watch the part as it evolves)
4. Save that part as a native solidworks part by automatically reconstructing smooth, editable NURBS/CAD surfaces rather than leaving you with a raw mesh.



Step 1. Determine what solving method results in the best outcome (can combine multiple solving methods that feed into eachother, or a single unified one)

Step 2. Compare our simulation of our final part to solidworks simulation of that part to confirm that our dynamics and solvers are correct so what is feeding the GSO is actually correct.

Step 3. Iterate on our code to create a better solver

Step 4. Create the code that automatically reconstructs smooth, editable NURBS/CAD surfaces rather than leaving you with a raw mesh.

Step 5. Create the user interactable Viewport that allows for the User to do these simulations themself in a familiar way to Solidworks simulation.



