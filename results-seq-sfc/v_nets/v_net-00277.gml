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
  id 277
  arrival_time 5355.977387915263
  lifetime 344.4532044053708
  num_nodes 4
  type "path"
  node [
    id 0
    label "0"
    cpu 27
    gpu 45
    rom 21
  ]
  node [
    id 1
    label "1"
    cpu 17
    gpu 0
    rom 13
  ]
  node [
    id 2
    label "2"
    cpu 2
    gpu 27
    rom 5
  ]
  node [
    id 3
    label "3"
    cpu 32
    gpu 9
    rom 14
  ]
  edge [
    source 0
    target 1
    bw 34
  ]
  edge [
    source 1
    target 2
    bw 15
  ]
  edge [
    source 2
    target 3
    bw 36
  ]
]
