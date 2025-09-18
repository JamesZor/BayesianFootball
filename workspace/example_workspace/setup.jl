module ExampleModule

  struct ExampleStruct
    var_one::Float64
    var_two::Int64
  end 

  function simple(obj::ExampleStruct)
    return obj.var_one + obj.var_two 
  end

end 
