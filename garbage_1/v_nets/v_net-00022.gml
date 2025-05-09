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
  id 22
  arrival_time 543.7431900856087
  lifetime 353.81861327189193
  num_nodes 8
  type "path"
  node [
    id 0
    label "0"
    cpu 2
    gpu 46
    rom 27
  ]
  node [
    id 1
    label "1"
    cpu 47
    gpu 41
    rom 1
  ]
  node [
    id 2
    label "2"
    cpu 41
    gpu 4
    rom 32
  ]
  node [
    id 3
    label "3"
    cpu 37
    gpu 46
    rom 6
  ]
  node [
    id 4
    label "4"
    cpu 8
    gpu 0
    rom 16
  ]
  node [
    id 5
    label "5"
    cpu 40
    gpu 20
    rom 22
  ]
  node [
    id 6
    label "6"
    cpu 27
    gpu 24
    rom 22
  ]
  node [
    id 7
    label "7"
    cpu 7
    gpu 34
    rom 45
  ]
  edge [
    source 0
    target 1
    bw 8
  ]
  edge [
    source 1
    target 2
    bw 31
  ]
  edge [
    source 2
    target 3
    bw 37
  ]
  edge [
    source 3
    target 4
    bw 39
  ]
  edge [
    source 4
    target 5
    bw 47
  ]
  edge [
    source 5
    target 6
    bw 9
  ]
  edge [
    source 6
    target 7
    bw 10
  ]
]
