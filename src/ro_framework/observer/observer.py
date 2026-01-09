"""
Observer implementation.

An observer is a configuration within the Block Universe characterized by:
- Boundary (internal/external DoF partition)
- Mapping functions (external → internal)
- Resolution (per-DoF finite granularity)
- Memory (correlation structure across temporal DoF)
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from ro_framework.core.dof import DoF
from ro_framework.core.state import State
from ro_framework.observer.mapping import MappingFunction


@dataclass
class Observer:
    """
    Observer: A configuration within the Block characterized by:
    - Boundary (B): Partition of DoFs into internal and external
    - Mapping (M): External → Internal function
    - Resolution (R): Per-DoF finite granularity
    - Memory (Mem): Correlation structure across temporal DoF

    Mathematical notation: O = (B, M, R, Mem)

    The observer maps external configurations to internal configurations,
    maintaining finite resolution and potentially memory structure.

    Attributes:
        name: Identifier for this observer
        internal_dofs: DoFs internal to the observer
        external_dofs: DoFs external to the observer
        world_model: Mapping from external to internal DoFs
        self_model: Optional mapping from internal to internal (for consciousness)
        resolution: Per-DoF resolution limits
        temporal_dof: Optional temporal DoF for memory tracking
        memory_buffer: Finite-length history of internal states
        internal_state: Current internal state

    Example:
        >>> # Define DoFs
        >>> external_dof = PolarDoF(name="sensor", pole_negative=-1, pole_positive=1)
        >>> internal_dof = PolarDoF(name="latent", pole_negative=-10, pole_positive=10)
        >>>
        >>> # Create mapping
        >>> world_model = IdentityMapping(
        ...     input_dofs=[external_dof],
        ...     output_dofs=[internal_dof]
        ... )
        >>>
        >>> # Create observer
        >>> observer = Observer(
        ...     name="simple_observer",
        ...     internal_dofs=[internal_dof],
        ...     external_dofs=[external_dof],
        ...     world_model=world_model
        ... )
        >>>
        >>> # Observe
        >>> external_state = State(values={external_dof: 0.5})
        >>> internal_state = observer.observe(external_state)
    """

    name: str
    internal_dofs: List[DoF]
    external_dofs: List[DoF]
    world_model: MappingFunction
    self_model: Optional[MappingFunction] = None
    resolution: Dict[DoF, float] = field(default_factory=dict)
    temporal_dof: Optional[DoF] = None
    memory_buffer: List[State] = field(default_factory=list)
    memory_capacity: int = 1000
    internal_state: Optional[State] = None

    def __post_init__(self) -> None:
        """Initialize resolution dict if empty."""
        if not self.resolution:
            self.resolution = {dof: 1e-6 for dof in self.internal_dofs}

    def observe(self, external_state: State) -> State:
        """
        Perform observation: map external DoFs to internal DoFs.

        This is the core mechanism of observation in the framework.
        The observer applies its world model to translate external
        sensory data into internal representations.

        Args:
            external_state: State with values on external DoFs

        Returns:
            Internal state with values on internal DoFs

        Example:
            >>> external = State(values={vision_dof: image_data})
            >>> internal = observer.observe(external)
        """
        # Apply world model mapping
        internal_state = self.world_model(external_state)

        # Update internal state
        self.internal_state = internal_state

        # Store in memory (correlation across temporal DoF)
        if self.temporal_dof is not None:
            self.memory_buffer.append(internal_state)

            # Maintain memory capacity
            if len(self.memory_buffer) > self.memory_capacity:
                self.memory_buffer.pop(0)

        return internal_state

    def self_observe(self) -> Optional[State]:
        """
        Perform self-observation: map internal DoFs to internal DoFs.

        This is the recursive self-modeling that defines consciousness
        in the structural sense. The observer applies its self-model to
        its own internal state.

        Returns:
            Self-representation state, or None if no self-model

        Example:
            >>> self_repr = observer.self_observe()
            >>> if self_repr is not None:
            ...     print(f"Observer is self-aware: {self_repr}")
        """
        if self.self_model is None:
            return None

        if self.internal_state is None:
            return None

        # Apply self-model mapping (recursion!)
        self_representation = self.self_model(self.internal_state)

        return self_representation

    def get_resolution(self, dof: DoF) -> float:
        """
        Get resolution limit for a specific DoF.

        Args:
            dof: DoF to get resolution for

        Returns:
            Resolution limit (minimum distinguishable difference)
        """
        return self.resolution.get(dof, 1e-6)

    def has_memory(self, threshold: float = 0.5, lag: int = 1) -> bool:
        """
        Check if observer has memory structure.

        Memory exists if correlation across temporal DoF exceeds
        what would be expected from instantaneous external correlations.

        Args:
            threshold: Minimum correlation to consider "significant"
            lag: Temporal lag for autocorrelation

        Returns:
            True if memory structure detected

        Example:
            >>> if observer.has_memory():
            ...     print("Observer has memory!")
        """
        if self.temporal_dof is None or len(self.memory_buffer) < lag + 2:
            return False

        # Check for temporal correlation in any internal DoF
        # This is a simplified check - full implementation would compute
        # autocorrelation for each DoF

        for dof in self.internal_dofs:
            # Extract values for this DoF across time
            values = []
            for state in self.memory_buffer:
                val = state.get_value(dof)
                if val is not None:
                    values.append(float(val))

            if len(values) < lag + 2:
                continue

            # Compute lagged correlation
            v1 = np.array(values[:-lag])
            v2 = np.array(values[lag:])

            if len(v1) > 1 and np.std(v1) > 0 and np.std(v2) > 0:
                corr = np.corrcoef(v1, v2)[0, 1]
                if abs(corr) > threshold:
                    return True

        return False

    def is_conscious(self) -> bool:
        """
        Check if observer has structural features of consciousness.

        Consciousness requires:
        - Self-model exists (recursive internal→internal mapping)
        - Self-model has same architectural type as world model
        - Achieves at least depth 1 recursion

        Returns:
            True if structural consciousness criteria met

        Example:
            >>> if observer.is_conscious():
            ...     print("Observer is structurally conscious!")
        """
        if self.self_model is None:
            return False

        # Check if both models have similar structure
        # In practice, this means checking if they're the same type
        # This is a simplified check - full implementation would verify
        # architectural similarity more rigorously

        return True  # If self_model exists, basic criterion is met

    def recursive_depth(self) -> int:
        """
        Compute depth of recursive self-modeling.

        - Depth 0: No self-model
        - Depth 1: Self-model exists
        - Depth 2+: Meta-models exist (model of modeling process)

        Returns:
            Recursive depth

        Example:
            >>> depth = observer.recursive_depth()
            >>> print(f"Recursive depth: {depth}")
        """
        if self.self_model is None:
            return 0

        # For now, return 1 if self-model exists
        # Full implementation would check for meta-meta-models
        return 1

    def know(
        self,
        external_dof: DoF,
        threshold: float = 0.7,
        min_samples: int = 10,
    ) -> bool:
        """
        Check if observer has knowledge of an external DoF.

        Knowledge requires:
        1. High correlation between external and internal DoFs
        2. Stability across contexts
        3. Bounded error (accuracy)
        4. Calibration (confidence matches accuracy)

        Args:
            external_dof: External DoF to check knowledge of
            threshold: Minimum correlation for "knowledge"
            min_samples: Minimum number of observations required

        Returns:
            True if knowledge criteria are met

        Example:
            >>> if observer.know(vision_dof):
            ...     print("Observer knows about vision!")
        """
        if len(self.memory_buffer) < min_samples:
            return False

        # This is a placeholder implementation
        # Full implementation would:
        # 1. Find which internal DoF(s) correlate with external_dof
        # 2. Compute correlation strength
        # 3. Check calibration
        # 4. Verify stability

        # For now, return False (unknown)
        return False

    def estimate_uncertainty(self, dof: DoF) -> float:
        """
        Estimate uncertainty in current knowledge of a DoF.

        Uncertainty comes from:
        - Resolution limits (structural)
        - Measurement noise (physical)
        - Model uncertainty (epistemic)

        Args:
            dof: DoF to estimate uncertainty for

        Returns:
            Uncertainty estimate

        Example:
            >>> uncertainty = observer.estimate_uncertainty(latent_dof)
            >>> print(f"Uncertainty: {uncertainty:.4f}")
        """
        if self.internal_state is None:
            return 1.0  # Maximum uncertainty

        # Get resolution-based uncertainty
        resolution_uncertainty = self.get_resolution(dof)

        # Add model uncertainty if available
        if hasattr(self.world_model, "compute_uncertainty"):
            model_uncertainty_dict = self.world_model.compute_uncertainty(self.internal_state)
            model_uncertainty = model_uncertainty_dict.get(dof, 0.0)
        else:
            model_uncertainty = 0.0

        # Combine uncertainties (simplified - should use proper uncertainty propagation)
        total_uncertainty = resolution_uncertainty + model_uncertainty

        return total_uncertainty

    def clear_memory(self) -> None:
        """
        Clear the memory buffer.

        Useful for resetting the observer or managing memory usage.

        Example:
            >>> observer.clear_memory()
        """
        self.memory_buffer.clear()

    def get_memory_size(self) -> int:
        """
        Get current size of memory buffer.

        Returns:
            Number of states in memory

        Example:
            >>> size = observer.get_memory_size()
            >>> print(f"Memory size: {size}")
        """
        return len(self.memory_buffer)

    def __repr__(self) -> str:
        """String representation of observer."""
        return (
            f"Observer(name='{self.name}', "
            f"internal_dofs={len(self.internal_dofs)}, "
            f"external_dofs={len(self.external_dofs)}, "
            f"conscious={self.is_conscious()}, "
            f"memory_size={self.get_memory_size()})"
        )
