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
  id 584
  arrival_time 10968.587192127201
  lifetime 56.97084180593055
  num_nodes 4
  type "path"
  node [
    id 0
    label "0"
    cpu 50
    gpu 39
    rom 31
  ]
  node [
    id 1
    label "1"
    cpu 43
    gpu 8
    rom 0
  ]
  node [
    id 2
    label "2"
    cpu 43
    gpu 44
    rom 4
  ]
  node [
    id 3
    label "3"
    cpu 45
    gpu 26
    rom 38
  ]
  edge [
    source 0
    target 1
    bw 27
  ]
  edge [
    source 1
    target 2
    bw 7
  ]
  edge [
    source 2
    target 3
    bw 40
  ]
]
