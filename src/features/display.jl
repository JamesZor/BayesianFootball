# src/features/display.jl

function Base.show(io::IO, ::MIME"text/plain", config::AbstractFeatureConfig)
    struct_name = string(nameof(typeof(config)))
    printstyled(io, "  $struct_name", color=:green, bold=true)
    
    props = propertynames(config)
    if isempty(props)
        println(io, "()")
        return
    end

    println(io, "(")
    for (i, prop) in enumerate(props)
        val = getproperty(config, prop)
        is_last = (i == length(props))
        prefix = is_last ? "    └── " : "    ├── "
        
        printstyled(io, prefix, color=:light_black)
        printstyled(io, string(prop), color=:white)
        printstyled(io, " = ", color=:light_black)
        
        if typeof(val) <: AbstractRatingTracker
            # Don't println, we will print it below
            println(io)
            # Add extra indent for the tracker
            str = sprint(show, MIME("text/plain"), val)
            # Indent all lines
            indented = join(["        " * line for line in split(str, "\n")], "\n")
            print(io, indented)
        else
            printstyled(io, string(val), color=:cyan)
            println(io)
        end
    end
    print(io, "  )")
end

function Base.show(io::IO, config::AbstractFeatureConfig)
    print(io, string(nameof(typeof(config))), "(...)")
end

function Base.show(io::IO, ::MIME"text/plain", t::AbstractRatingTracker)
    struct_name = string(nameof(typeof(t)))
    printstyled(io, "$struct_name\n", color=:green, bold=true)
    
    props = propertynames(t)
    for (i, prop) in enumerate(props)
        val = getproperty(t, prop)
        is_last = (i == length(props))
        prefix = is_last ? "└── " : "├── "
        
        printstyled(io, prefix, color=:light_black)
        printstyled(io, string(prop), color=:white)
        printstyled(io, ": ", color=:light_black)
        printstyled(io, string(val), color=:cyan)
        if !is_last
            println(io)
        end
    end
end

function Base.show(io::IO, t::AbstractRatingTracker)
    print(io, string(nameof(typeof(t))), "(...)")
end
