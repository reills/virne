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
  id 9
  arrival_time 151.03861508455051
  lifetime 1417.086079124515
  num_nodes 10
  type "path"
  node [
    id 0
    label "0"
    cpu 40
    gpu 42
    rom 5
  ]
  node [
    id 1
    label "1"
    cpu 40
    gpu 13
    rom 19
  ]
  node [
    id 2
    label "2"
    cpu 41
    gpu 34
    rom 14
  ]
  node [
    id 3
    label "3"
    cpu 11
    gpu 8
    rom 36
  ]
  node [
    id 4
    label "4"
    cpu 1
    gpu 15
    rom 1
  ]
  node [
    id 5
    label "5"
    cpu 14
    gpu 22
    rom 21
  ]
  node [
    id 6
    label "6"
    cpu 7
    gpu 16
    rom 37
  ]
  node [
    id 7
    label "7"
    cpu 45
    gpu 14
    rom 25
  ]
  node [
    id 8
    label "8"
    cpu 6
    gpu 50
    rom 11
  ]
  node [
    id 9
    label "9"
    cpu 3
    gpu 2
    rom 28
  ]
  edge [
    source 0
    target 1
    bw 45
  ]
  edge [
    source 1
    target 2
    bw 37
  ]
  edge [
    source 2
    target 3
    bw 11
  ]
  edge [
    source 3
    target 4
    bw 33
  ]
  edge [
    source 4
    target 5
    bw 42
  ]
  edge [
    source 5
    target 6
    bw 15
  ]
  edge [
    source 6
    target 7
    bw 48
  ]
  edge [
    source 7
    target 8
    bw 27
  ]
  edge [
    source 8
    target 9
    bw 4
  ]
]
