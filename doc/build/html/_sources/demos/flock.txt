The Flock demo
==============

This program implements a demonstration of the action of a Particle Swarm
Optimizer using the :ref:`pso` package. To run it, you must have the `PyQt4
<http://www.riverbankcomputing.co.uk/>`_ library installed. The demo shows a
small window representing a two-dimensional space, with a red dot being the
goal, and several white dots representing the particles. The program animates
the swarm searching for the goal.

Starting the Program
--------------------

Before trying to run the program, please make sure that the required libraries
are installed and working. Usually, they are not difficult to install: Windows
builds come with an installer, and Linux versions can be installed using a
package manager.

The program resides in the ``demos\flock`` folder. If you use Windows, use the
file browser to navigate to that folder and double-click the ``IPApp.py`` file,
and the program should start. In Linux, you can start it by the command line
interface, navigating to the program folder and typing::

    # python FlockApp.py

The main window of the program will be shown as in the picture below.

.. image:: figs/flock.png
    :width: 700


Controlling
-----------

The program itself is very simple. In the left side of the window, there is a
black field representing some space. In it, the red dot represents the goal, the
optimal point to be found by the swarm (internally, it is represented as the
maximum point in a quadratic function). White points represent each individual
in the swarm. At each iteration of the algorithm, the position of the white dots
is updated in the visualization space to reflect the search. After a fixed 
interval, the spot is randomly moved to another point and the swarm is reset.

In the right side of the window, there are control buttons, explained below:

    Start
        Starts the animation. When the animation is running, this button is
        made inactive.

    Stop
        Stops the animation if it is running. This button is normally inactive,
        but is made active when the animation is running.

    Step
        Performs a single step of the control: given the values of the angular
        position and speed, calculates the force that should be applied to the
        cart and updates the view. This button is made inactive when the
        animation is running.

    Change Spot
        By pressing this button, the red spot representing the search goal is
        immediatly moved to another randomly selected point.

    Reset
        With the activation of this command, every parameter in the simulation
        is randomly modified: the goal spot is moved to another position and a
        new swarm is generated.

    Delay (ms)
        The simulation waits a small amount of time between each iteration, to
        make the animation smoother and help visualizing what is actually
        happening. If you change this value, this interval is changed. Be aware
        that the wait time is given in miliseconds.
