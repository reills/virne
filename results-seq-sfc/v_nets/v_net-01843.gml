graph [
  node_attrs_setting [
    name "cpu"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  node_attrs_setting [
    name "gpu"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  node_attrs_setting [
    name "rom"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "node"
    type "resource"
  ]
  link_attrs_setting "_networkx_list_start"
  link_attrs_setting [
    name "bw"
    distribution "uniform"
    dtype "int"
    generative 1
    low 0
    high 50
    owner "link"
    type "resource"
  ]
  id 1843
  arrival_time 40702.097223682664
  lifetime 78.93503054927055
  num_nodes 2
  type "path"
  node [
    id 0
    label "0"
    cpu 10
    gpu 37
    rom 1
  ]
  node [
    id 1
    label "1"
    cpu 49
    gpu 0
    rom 23
  ]
  edge [
    source 0
    target 1
    bw 29
  ]
]
