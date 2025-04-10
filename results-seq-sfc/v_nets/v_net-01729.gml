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
  id 1729
  arrival_time 38560.679246163694
  lifetime 839.0836606945663
  num_nodes 14
  type "path"
  node [
    id 0
    label "0"
    cpu 29
    gpu 13
    rom 4
  ]
  node [
    id 1
    label "1"
    cpu 6
    gpu 0
    rom 17
  ]
  node [
    id 2
    label "2"
    cpu 29
    gpu 1
    rom 26
  ]
  node [
    id 3
    label "3"
    cpu 35
    gpu 14
    rom 2
  ]
  node [
    id 4
    label "4"
    cpu 29
    gpu 26
    rom 41
  ]
  node [
    id 5
    label "5"
    cpu 39
    gpu 42
    rom 11
  ]
  node [
    id 6
    label "6"
    cpu 11
    gpu 19
    rom 0
  ]
  node [
    id 7
    label "7"
    cpu 21
    gpu 25
    rom 50
  ]
  node [
    id 8
    label "8"
    cpu 39
    gpu 7
    rom 4
  ]
  node [
    id 9
    label "9"
    cpu 33
    gpu 49
    rom 17
  ]
  node [
    id 10
    label "10"
    cpu 12
    gpu 6
    rom 7
  ]
  node [
    id 11
    label "11"
    cpu 21
    gpu 19
    rom 31
  ]
  node [
    id 12
    label "12"
    cpu 20
    gpu 49
    rom 4
  ]
  node [
    id 13
    label "13"
    cpu 39
    gpu 3
    rom 3
  ]
  edge [
    source 0
    target 1
    bw 8
  ]
  edge [
    source 1
    target 2
    bw 15
  ]
  edge [
    source 2
    target 3
    bw 38
  ]
  edge [
    source 3
    target 4
    bw 45
  ]
  edge [
    source 4
    target 5
    bw 19
  ]
  edge [
    source 5
    target 6
    bw 49
  ]
  edge [
    source 6
    target 7
    bw 27
  ]
  edge [
    source 7
    target 8
    bw 49
  ]
  edge [
    source 8
    target 9
    bw 13
  ]
  edge [
    source 9
    target 10
    bw 49
  ]
  edge [
    source 10
    target 11
    bw 15
  ]
  edge [
    source 11
    target 12
    bw 3
  ]
  edge [
    source 12
    target 13
    bw 36
  ]
]
