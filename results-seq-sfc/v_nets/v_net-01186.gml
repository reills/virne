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
  id 1186
  arrival_time 24593.896777738773
  lifetime 3512.7049901756627
  num_nodes 8
  type "path"
  node [
    id 0
    label "0"
    cpu 23
    gpu 6
    rom 15
  ]
  node [
    id 1
    label "1"
    cpu 12
    gpu 32
    rom 23
  ]
  node [
    id 2
    label "2"
    cpu 44
    gpu 34
    rom 13
  ]
  node [
    id 3
    label "3"
    cpu 31
    gpu 41
    rom 48
  ]
  node [
    id 4
    label "4"
    cpu 41
    gpu 1
    rom 6
  ]
  node [
    id 5
    label "5"
    cpu 40
    gpu 27
    rom 36
  ]
  node [
    id 6
    label "6"
    cpu 33
    gpu 16
    rom 48
  ]
  node [
    id 7
    label "7"
    cpu 3
    gpu 15
    rom 35
  ]
  edge [
    source 0
    target 1
    bw 9
  ]
  edge [
    source 1
    target 2
    bw 39
  ]
  edge [
    source 2
    target 3
    bw 23
  ]
  edge [
    source 3
    target 4
    bw 38
  ]
  edge [
    source 4
    target 5
    bw 3
  ]
  edge [
    source 5
    target 6
    bw 11
  ]
  edge [
    source 6
    target 7
    bw 19
  ]
]
