# src/Models/PreGame/types.jl

# ==========================================
# 1. ABSTRACT COMPONENT INTERFACES
# ==========================================
abstract type AbstractDispersionConfig end
abstract type AbstractHomeAdvantageConfig end
abstract type AbstractKappaConfig end
abstract type AbstractDynamicsConfig end
abstract type AbstractInterceptionConfig end

# ==========================================
# 2. MASTER ARCHITECTURE TYPES
# ==========================================
# Engine model config structs have been moved to their respective engine files 
# in src/Models/PreGame/engines/ to keep them self-contained.
