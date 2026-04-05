# Mechanisms for Coherent Phonon Generation

Coherent phonons are the vibration of the structural lattice in a material. Phonons can be detected using infrared and Raman (visible) spectroscopy. For more accurate measurement of the phonon's oscillatory frequency and their natural damping, ultrafast laser (pump-probe) spectroscopy is used. This is because these pump-probe systems can measure in the same time scale (femtoseconds) that these vibrations occur at.

We can detect phonons by observing periodic changes in the physical properties of the material. Here we observe how the material's refractive index changes. The phonons are coherent in the sense that their atomic motions are perfectly synchronous in the entire photoexcited volume. This is the volume of the material that the pump beam excites. The probe beam must target a smaller area than the pump beam excites. Otherwise, the non-excited areas skew our results.

These phonons exhibit driven damped harmonic motion. This motion is generated through two distinct mechanisms:
1. Impulsive Stimulated Raman Scattering (ISRS)
2. Displacive Excitation of Coherent Phonons (DECP)

The general equation of motion for the phonons is:
$$ \frac{\partial^2Q(t)}{\partial t^2} + 2\gamma_{\text{ph}}\frac{\partial Q(t)}{\partial t} + \omega^2_{\text{ph}}Q(t) = F^Q(t) $$

**ISRS** is an excitation mechanism in which two pump photons with slightly different energies
$\omega_1$, $\omega_2$ stimulate the Raman excitation of a phonon with energy $\omega_{\text{ph}} = \omega_{1} - \omega_{2} $. The phonon generation is impulsive because it is initiated by a pump pulse with a duration $\Delta t$ shorter than the phonon period $T$ and the pump can be approximated with a $\delta(t) function. The equation of motion becomes: 
$$ \frac{\partial^2Q(t)}{\partial t^2} + 2\gamma_{\text{ph}}\frac{\partial Q(t)}{\partial t} + \omega^2_{\text{ph}}Q(t) = E_0\delta(t) $$
Where $E_0$ is the amplitude of the pump electric field. The ISRS solution is a damped sine oscillation, where $\Theta(t)$ is the Heavyside step function:
$$ Q(t) \propto \Theta(t)e^{-\gamma t}\sin{\left[\sqrt{\omega_{0}^2-\gamma^2}t\right]} $$
ISRS can be observed both in transparent(insulating) and opaque(semiconducting and metallic) materials. Antimony and Bismuth are somewhere between these. ISRS only involes Raman active modes and is primarily driven by the pump's electric field.

Raman active modes are molecular vibrations or rotations that cause a change in the polarizability of a molecule's electron cloud, enabling them to scatter incident light and appear as peaks in a Raman spectrum.

**DECP** is an excitation where the pump excites electrons from the ground state, but the nuclei stay in their equilibrium positions. The electrons thermalize (reach thermal equilibrium) in their new conduction band minimum. This produces an electronic distribution distinct from the equilibrium one.

This new distribution produces a new electrostatic environment for the nuclei. The nuclei react by moving into new coordinates to minimize the system's total free energy. This displacement occurs along totally-symmetric Raman coordinates with no change in lattice symmetry. It follows a damped oscillatory behavior along the new lattice positions. The recombination of the conduction band and the holes in the valance band (electron return to ground state) is a much slower process that occurs on a nanosecond time scale. As such the new environment can be treated as constant when compared to the phonon period.

This new environment is the driving force for our motion and can be approximated by:
$$ F^Q(t) = \kappa \Delta n \Theta(t) $$
$\kappa$ is the electron-phonon coupling constant and $\Delta n$ is the change in electronic density. Our new equation of motion is:
$$ \frac{\partial^2Q(t)}{\partial t^2} + 2\gamma_{\text{ph}}\frac{\partial Q(t)}{\partial t} + \omega^2_{\text{ph}}Q(t) = \kappa \Delta n \Theta(t) $$
The DECP solution is a damped cosine oscillation about the new equilibrium position:
$$ Q(t) \propto -\frac{\kappa \Delta n}{\omega_{0}^2}e^{-\gamma t}\cos{\left[\sqrt{\omega_{0}^2-\gamma^2}t\right]} + \frac{\kappa \Delta n}{\omega_{0}^2} = \frac{\kappa \Delta n}{\omega_{0}^2}(1 - e^{-\gamma t}\cos{\left[\sqrt{\omega_{0}^2-\gamma^2}t\right]}) $$

Our most accurate solution will likely be some combination of these two:
$$ Q(t) = A\Theta(t)e^{-\gamma t}\sin{\left[\sqrt{\omega_{0}^2-\gamma^2}t\right]} + B\frac{\kappa \Delta n}{\omega_{0}^2}\left(1 - e^{-\gamma t}\cos{\left[\sqrt{\omega_{0}^2-\gamma^2}t\right]}\right) $$
Where $A$ and $B$ are some constants.

