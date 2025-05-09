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
  id 1088
  arrival_time 22708.712176821766
  lifetime 421.27622599674476
  num_nodes 9
  type "path"
  node [
    id 0
    label "0"
    cpu 32
    gpu 27
    rom 50
  ]
  node [
    id 1
    label "1"
    cpu 50
    gpu 5
    rom 28
  ]
  node [
    id 2
    label "2"
    cpu 46
    gpu 23
    rom 0
  ]
  node [
    id 3
    label "3"
    cpu 43
    gpu 46
    rom 2
  ]
  node [
    id 4
    label "4"
    cpu 5
    gpu 3
    rom 38
  ]
  node [
    id 5
    label "5"
    cpu 38
    gpu 49
    rom 26
  ]
  node [
    id 6
    label "6"
    cpu 22
    gpu 40
    rom 42
  ]
  node [
    id 7
    label "7"
    cpu 7
    gpu 20
    rom 31
  ]
  node [
    id 8
    label "8"
    cpu 17
    gpu 3
    rom 35
  ]
  edge [
    source 0
    target 1
    bw 5
  ]
  edge [
    source 1
    target 2
    bw 48
  ]
  edge [
    source 2
    target 3
    bw 2
  ]
  edge [
    source 3
    target 4
    bw 41
  ]
  edge [
    source 4
    target 5
    bw 28
  ]
  edge [
    source 5
    target 6
    bw 24
  ]
  edge [
    source 6
    target 7
    bw 12
  ]
  edge [
    source 7
    target 8
    bw 10
  ]
]
