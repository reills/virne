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
  id 1635
  arrival_time 36642.314109827894
  lifetime 260.44552896218113
  num_nodes 5
  type "path"
  node [
    id 0
    label "0"
    cpu 30
    gpu 28
    rom 9
  ]
  node [
    id 1
    label "1"
    cpu 8
    gpu 15
    rom 48
  ]
  node [
    id 2
    label "2"
    cpu 11
    gpu 20
    rom 26
  ]
  node [
    id 3
    label "3"
    cpu 49
    gpu 46
    rom 21
  ]
  node [
    id 4
    label "4"
    cpu 24
    gpu 17
    rom 10
  ]
  edge [
    source 0
    target 1
    bw 2
  ]
  edge [
    source 1
    target 2
    bw 9
  ]
  edge [
    source 2
    target 3
    bw 43
  ]
  edge [
    source 3
    target 4
    bw 43
  ]
]
