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
  id 906
  arrival_time 19344.92726883893
  lifetime 352.0412859669366
  num_nodes 5
  type "path"
  node [
    id 0
    label "0"
    cpu 42
    gpu 32
    rom 40
  ]
  node [
    id 1
    label "1"
    cpu 26
    gpu 14
    rom 0
  ]
  node [
    id 2
    label "2"
    cpu 20
    gpu 8
    rom 34
  ]
  node [
    id 3
    label "3"
    cpu 2
    gpu 4
    rom 5
  ]
  node [
    id 4
    label "4"
    cpu 36
    gpu 1
    rom 3
  ]
  edge [
    source 0
    target 1
    bw 16
  ]
  edge [
    source 1
    target 2
    bw 14
  ]
  edge [
    source 2
    target 3
    bw 38
  ]
  edge [
    source 3
    target 4
    bw 13
  ]
]
