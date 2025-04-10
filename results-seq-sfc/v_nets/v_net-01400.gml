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
  id 1400
  arrival_time 29396.342852302045
  lifetime 3.9399535603321727
  num_nodes 10
  type "path"
  node [
    id 0
    label "0"
    cpu 28
    gpu 22
    rom 42
  ]
  node [
    id 1
    label "1"
    cpu 36
    gpu 1
    rom 41
  ]
  node [
    id 2
    label "2"
    cpu 29
    gpu 50
    rom 45
  ]
  node [
    id 3
    label "3"
    cpu 10
    gpu 48
    rom 5
  ]
  node [
    id 4
    label "4"
    cpu 21
    gpu 33
    rom 15
  ]
  node [
    id 5
    label "5"
    cpu 0
    gpu 11
    rom 23
  ]
  node [
    id 6
    label "6"
    cpu 23
    gpu 13
    rom 6
  ]
  node [
    id 7
    label "7"
    cpu 35
    gpu 8
    rom 2
  ]
  node [
    id 8
    label "8"
    cpu 50
    gpu 8
    rom 40
  ]
  node [
    id 9
    label "9"
    cpu 9
    gpu 10
    rom 44
  ]
  edge [
    source 0
    target 1
    bw 10
  ]
  edge [
    source 1
    target 2
    bw 44
  ]
  edge [
    source 2
    target 3
    bw 1
  ]
  edge [
    source 3
    target 4
    bw 44
  ]
  edge [
    source 4
    target 5
    bw 29
  ]
  edge [
    source 5
    target 6
    bw 33
  ]
  edge [
    source 6
    target 7
    bw 47
  ]
  edge [
    source 7
    target 8
    bw 39
  ]
  edge [
    source 8
    target 9
    bw 5
  ]
]
